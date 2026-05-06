# SPDX-License-Identifier: Apache-2.0
"""Benchmark: EloqStoreBackend vs LocalDiskBackend put/get throughput.

Each chunk is a single-layer KV cache tensor, shape [256, 2, 128] @ float16 = 128 KB.
This is the realistic per-chunk size in LMCache (chunk_size=256 tokens).
"""
# Standard
import argparse
import asyncio
import shutil
import tempfile
import threading
import time
from typing import List, Sequence

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

# ── Config ─────────────────────────────────────────────────────────
DEFAULT_N_CHUNKS = 1024
DEFAULT_WARMUP_CHUNKS = 16
CHUNK_SHAPE = torch.Size([256, 2, 128])  # 128 KB @ float16
DTYPE = torch.float16
FMT = MemoryFormat.KV_2LTD
CAPACITY_GB = 2

CHUNK_BYTES = CHUNK_SHAPE.numel() * DTYPE.itemsize  # 128 KB


def _make_config(disk_path: str) -> LMCacheEngineConfig:
    return LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.5,
        local_disk=disk_path,
        max_local_disk_size=CAPACITY_GB,
    )


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="bm",
        world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0,
        kv_dtype=DTYPE, kv_shape=(1, 2, 256, 8, 128),
        role="worker", chunk_size=256,
    )


def _make_keys(n: int) -> List[CacheEngineKey]:
    return [
        CacheEngineKey("bm", 1, 0, i, DTYPE)
        for i in range(n)
    ]


class BenchLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.t.start()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.t.join(timeout=5)


class CompletionTracker:
    def __init__(self, expected: int):
        self.expected = expected
        self.completed = 0
        self._event = threading.Event()
        self._lock = threading.Lock()

    def callback(self, _key: CacheEngineKey) -> None:
        with self._lock:
            self.completed += 1
            if self.completed >= self.expected:
                self._event.set()

    def wait(self, timeout: float) -> None:
        if self.expected == 0:
            return
        if not self._event.wait(timeout):
            raise TimeoutError(
                f"Timed out waiting for writes: completed={self.completed}, "
                f"expected={self.expected}"
            )


def _allocate_objects(cpu_backend, count: int) -> List[MemoryObj]:
    objs = [cpu_backend.allocate(CHUNK_SHAPE, DTYPE, FMT) for _ in range(count)]
    for obj in objs:
        assert obj is not None
        tensor = obj.tensor
        assert tensor is not None
        tensor.uniform_(-1, 1)
    return objs


def _release_objects(objs: Sequence[MemoryObj]) -> None:
    for obj in objs:
        obj.ref_count_down()


def _wait_for_put_completion(
    backend,
    keys: Sequence[CacheEngineKey],
    objs: Sequence[MemoryObj],
    timeout: float,
) -> None:
    ordered_pairs = sorted(
        zip(keys, objs, strict=True),
        key=lambda item: item[0].to_string(),
    )
    ordered_keys = [key for key, _ in ordered_pairs]
    ordered_objs = [obj for _, obj in ordered_pairs]
    tracker = CompletionTracker(len(ordered_keys))
    futs = backend.batched_submit_put_task(
        ordered_keys,
        ordered_objs,
        on_complete_callback=tracker.callback,
    )
    if futs:
        for fut in futs:
            fut.result(timeout=timeout)
    tracker.wait(timeout=timeout)


def _read_all_serial(backend, keys: Sequence[CacheEngineKey]) -> List[MemoryObj]:
    results = [backend.get_blocking(key) for key in keys]
    missing = [key for key, result in zip(keys, results, strict=True) if result is None]
    if missing:
        raise AssertionError(f"Missing {len(missing)} keys during read: {missing[:4]}")
    return results


def run_bench(
    backend,
    warmup_keys: Sequence[CacheEngineKey],
    warmup_objs: Sequence[MemoryObj],
    bench_keys: Sequence[CacheEngineKey],
    bench_objs: Sequence[MemoryObj],
    label: str,
    timeout: float,
) -> None:
    # Warmup on a disjoint key set so measured writes remain cold inserts.
    _wait_for_put_completion(backend, warmup_keys, warmup_objs, timeout=timeout)

    # ── Write ──
    t0 = time.perf_counter()
    _wait_for_put_completion(backend, bench_keys, bench_objs, timeout=timeout)
    w_elapsed = time.perf_counter() - t0
    w_total = len(bench_objs) * CHUNK_BYTES
    w_mbps = (w_total / (1024 * 1024)) / w_elapsed

    # ── Read ──
    t0 = time.perf_counter()
    results = _read_all_serial(backend, bench_keys)
    total_bytes = sum(result.get_physical_size() for result in results)
    r_elapsed = time.perf_counter() - t0
    r_mbps = (total_bytes / (1024 * 1024)) / r_elapsed
    _release_objects(results)

    print(f"  {label:12s}  write {w_mbps:8.1f} MB/s   read {r_mbps:8.1f} MB/s  "
          f"({len(bench_keys)} × {CHUNK_BYTES//1024} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EloqStoreBackend vs LocalDiskBackend with aligned semantics."
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_N_CHUNKS,
        help="Number of chunks in the measured run.",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=DEFAULT_WARMUP_CHUNKS,
        help="Number of disjoint chunks written before the measured run.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-phase timeout when waiting for asynchronous writes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eloq_path = tempfile.mkdtemp(prefix="bm_eloq_")
    disk_path = tempfile.mkdtemp(prefix="bm_disk_")
    metadata = _make_metadata()
    total_keys = args.warmup_chunks + args.num_chunks
    keys = _make_keys(total_keys)
    warmup_keys = keys[:args.warmup_chunks]
    bench_keys = keys[args.warmup_chunks:]

    print(
        f"Warmup: {args.warmup_chunks} × {CHUNK_BYTES//1024} KB = "
        f"{args.warmup_chunks * CHUNK_BYTES / (1024 * 1024):.1f} MB"
    )
    print(
        f"Bench:  {args.num_chunks} × {CHUNK_BYTES//1024} KB = "
        f"{args.num_chunks * CHUNK_BYTES / (1024 * 1024):.1f} MB total\n"
    )

    # ── EloqStore ──────────────────────────────────────────────────
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.eloqstore_backend import EloqStoreBackend

    bm1 = BenchLoop()
    cfg1 = _make_config(disk_path)
    cfg1.extra_config = {
        "eloqstore.store_path": eloq_path,
        "eloqstore.table_name": "bm",
        "eloqstore.num_threads": 4,
    }
    cpu1 = LocalCPUBackend(cfg1, metadata, "cpu", lmcache_worker=None)
    be1 = EloqStoreBackend(dst_device="cpu", config=cfg1, metadata=metadata,
                            local_cpu_backend=cpu1, loop=bm1.loop)
    warmup_objs1 = _allocate_objects(cpu1, args.warmup_chunks)
    bench_objs1 = _allocate_objects(cpu1, args.num_chunks)
    run_bench(
        be1,
        warmup_keys,
        warmup_objs1,
        bench_keys,
        bench_objs1,
        "EloqStore",
        timeout=args.timeout_seconds,
    )
    _release_objects(warmup_objs1)
    _release_objects(bench_objs1)
    be1.close()
    bm1.stop()

    # ── LocalDisk ──────────────────────────────────────────────────
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

    bm2 = BenchLoop()
    cfg2 = _make_config(disk_path)
    cpu2 = LocalCPUBackend(cfg2, metadata, "cpu", lmcache_worker=None)
    be2 = LocalDiskBackend(config=cfg2, loop=bm2.loop, local_cpu_backend=cpu2,
                           dst_device="cpu", lmcache_worker=None, metadata=metadata)
    warmup_objs2 = _allocate_objects(cpu2, args.warmup_chunks)
    bench_objs2 = _allocate_objects(cpu2, args.num_chunks)
    run_bench(
        be2,
        warmup_keys,
        warmup_objs2,
        bench_keys,
        bench_objs2,
        "LocalDisk",
        timeout=args.timeout_seconds,
    )
    _release_objects(warmup_objs2)
    _release_objects(bench_objs2)
    be2.close()
    bm2.stop()

    shutil.rmtree(eloq_path, ignore_errors=True)
    shutil.rmtree(disk_path, ignore_errors=True)


if __name__ == "__main__":
    main()
