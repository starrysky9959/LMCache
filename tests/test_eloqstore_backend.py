# SPDX-License-Identifier: Apache-2.0
"""Smoke test for EloqStoreBackend — put/get, persistence, pin/unpin, remove."""
# Standard
import asyncio
import struct
import tempfile
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.eloqstore_backend import (
    _deserialize_value,
    _serialize_value,
    EloqStoreBackend,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def _make_config(store_path: str) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        max_local_disk_size=0.5,
    )
    config.extra_config = {
        "eloqstore.store_path": store_path,
        "eloqstore.table_name": "lmcache_smoke_test",
        "eloqstore.num_threads": 1,
    }
    return config


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(32, 2, 256, 8, 128),
        role="worker",
        chunk_size=256,
    )


def _make_key(chunk_hash: int = 0xABCD) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


class TestHarness:
    """Manages the event loop thread and backend lifecycle."""

    def __init__(self, tmpdir: str):
        self.tmpdir = tmpdir
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def create_backend(self) -> EloqStoreBackend:
        config = _make_config(self.tmpdir)
        metadata = _make_metadata()
        local_cpu = LocalCPUBackend(config, metadata, "cpu", lmcache_worker=None)
        return EloqStoreBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=self.loop,
        )

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def _release_if_present(obj) -> None:
    if obj is not None:
        obj.ref_count_down()


def test_serialize_round_trip():
    """Unit test for the binary header serde."""
    meta = {"shape": [256, 2, 128], "dtype": "float16", "fmt": 1, "size": 131072}
    payload = b"\x00" * 131072
    packed = _serialize_value(meta, payload)
    assert packed[:4] == struct.pack("<I", 0x454C4D43)
    result = _deserialize_value(packed)
    assert result is not None
    meta2, payload2 = result
    assert meta2 == meta
    assert payload2 == payload


def test_basic_round_trip():
    """Put a KV chunk, get it back, verify data integrity."""
    tmpdir = tempfile.mkdtemp(prefix="eloqstore_smoke_")
    h = TestHarness(tmpdir)
    try:
        backend = h.create_backend()

        shape = torch.Size([256, 2, 128])
        dtype = torch.float16
        fmt = MemoryFormat.KV_2LTD
        mem_obj = backend.local_cpu_backend.allocate(shape, dtype, fmt)
        assert mem_obj is not None
        tensor = mem_obj.tensor
        assert tensor is not None
        tensor.fill_(0.5)

        key = _make_key()

        # Before put
        assert not backend.contains(key, pin=False)

        # Put
        fut = backend.submit_put_task(key, mem_obj)
        if fut is not None:
            fut.result(timeout=10)

        # After put
        assert backend.contains(key, pin=False)

        # Get
        result = backend.get_blocking(key)
        assert result is not None
        rt = result.tensor
        assert rt is not None
        assert torch.allclose(rt, tensor, atol=1e-3), "Data mismatch"

        # Pin / unpin
        assert backend.pin(key)
        meta = backend.dict.get(key)
        assert meta is not None and meta.is_pinned
        assert backend.unpin(key)
        assert not meta.is_pinned

        # Remove
        assert backend.remove(key)
        assert not backend.contains(key, pin=False)
        assert backend.get_blocking(key) is None

        _release_if_present(result)
        _release_if_present(mem_obj)
        backend.close()
    finally:
        h.stop()


def test_index_persistence():
    """Verify that metadata survives close + reopen."""
    tmpdir = tempfile.mkdtemp(prefix="eloqstore_persist_")
    key = _make_key()

    # First session: put data, close (saves index)
    h1 = TestHarness(tmpdir)
    try:
        b1 = h1.create_backend()
        shape = torch.Size([256, 2, 128])
        dtype = torch.float16
        fmt = MemoryFormat.KV_2LTD
        mem_obj = b1.local_cpu_backend.allocate(shape, dtype, fmt)
        assert mem_obj is not None
        tensor = mem_obj.tensor
        assert tensor is not None
        tensor.fill_(0.75)

        fut = b1.submit_put_task(key, mem_obj)
        if fut is not None:
            fut.result(timeout=10)
        assert b1.contains(key)
        _release_if_present(mem_obj)
        b1.close()
    finally:
        h1.stop()

    # Second session: reopen, verify key is still there
    h2 = TestHarness(tmpdir)
    try:
        b2 = h2.create_backend()
        # Index should be restored; key must be present
        assert b2.contains(key), "Key not restored after reopen"
        result = b2.get_blocking(key)
        assert result is not None, "get_blocking returned None after reopen"
        rt = result.tensor
        assert rt is not None
        assert torch.allclose(rt, tensor, atol=1e-3), "Data mismatch after reopen"
        _release_if_present(result)
        b2.close()
    finally:
        h2.stop()


if __name__ == "__main__":
    test_serialize_round_trip()
    test_basic_round_trip()
    test_index_persistence()
    print("All smoke tests passed!")
