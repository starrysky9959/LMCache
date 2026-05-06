# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
import asyncio
import json
import os
import struct
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.native_clients.eloqstore_client import (
    EloqStoreClient,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.storage_backend import LocalCPUBackend

logger = init_logger(__name__)

# Binary header: 4B magic + 4B meta_len (uint32 LE)
_HEADER_MAGIC = 0x454C4D43  # "ELMC"
_HEADER_STRUCT = struct.Struct("<II")  # magic, meta_len
_INDEX_KEY = "__lmcache_index__"


def _serialize_value(meta: Dict[str, Any], payload: bytes) -> bytes:
    """Pack metadata JSON + payload into a single bytes buffer.

    Format: ``[4B magic][4B meta_len][meta_json_bytes][payload]``
    """
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = _HEADER_STRUCT.pack(_HEADER_MAGIC, len(meta_json))
    return header + meta_json + payload


def _deserialize_value(raw: bytes) -> Optional[tuple[Dict[str, Any], bytes]]:
    """Unpack a header-prefixed value into (meta_dict, payload).

    Returns ``None`` if the magic or header is invalid.
    """
    if len(raw) < _HEADER_STRUCT.size:
        return None
    magic, meta_len = _HEADER_STRUCT.unpack(raw[:_HEADER_STRUCT.size])
    if magic != _HEADER_MAGIC:
        return None
    meta_start = _HEADER_STRUCT.size
    meta_end = meta_start + meta_len
    if meta_end > len(raw):
        return None
    meta = json.loads(raw[meta_start:meta_end].decode("utf-8"))
    payload = raw[meta_end:]
    return meta, payload


def _meta_to_dict(
    shape: torch.Size,
    dtype: torch.dtype,
    fmt: MemoryFormat,
    size: int,
) -> Dict[str, Any]:
    return {
        "shape": list(shape),
        "dtype": TORCH_DTYPE_TO_STR_DTYPE.get(dtype, str(dtype)),
        "fmt": fmt.value,
        "size": size,
    }


def _meta_from_dict(meta: Dict[str, Any]) -> tuple[torch.Size, torch.dtype, MemoryFormat, int]:
    shape = torch.Size(meta["shape"])
    dtype = STR_DTYPE_TO_TORCH_DTYPE.get(meta["dtype"], torch.float16)
    fmt = MemoryFormat(meta["fmt"])
    size = meta["size"]
    return shape, dtype, fmt, size


class EloqStoreWorker:
    """Manages put tasks and async executor for EloqStoreBackend."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey) -> None:
        with self.put_lock:
            self.put_tasks.discard(key)

    def insert_put_task(self, key: CacheEngineKey) -> None:
        with self.put_lock:
            self.put_tasks.add(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class EloqStoreBackend(StoragePluginInterface):
    """Storage backend backed by EloqStore, an embedded local key-value store.

    Follows the same structure as ``LocalDiskBackend`` but stores KV chunks
    as bytes in EloqStore instead of as files on disk.

    Each value is stored with a binary header that includes serialized
    metadata (shape, dtype, fmt, size) so that individual entries are
    self-describing.  A global index is persisted under a well-known key
    to enable in-memory dict recovery across restarts.
    """

    def __init__(
        self,
        dst_device: str = "cuda",
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        if torch.cuda.is_available():
            super().__init__(dst_device=dst_device, config=config, metadata=metadata,
                             local_cpu_backend=local_cpu_backend, loop=loop)
        else:
            super().__init__(dst_device="cpu", config=config, metadata=metadata,
                             local_cpu_backend=local_cpu_backend, loop=loop)

        assert config is not None, "EloqStoreBackend requires a config"
        assert config.extra_config is not None, (
            "EloqStoreBackend requires extra_config with eloqstore.* keys"
        )
        assert local_cpu_backend is not None, (
            "EloqStoreBackend requires a LocalCPUBackend for memory allocation"
        )
        assert loop is not None, "EloqStoreBackend requires an event loop"

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.local_cpu_backend = local_cpu_backend
        self.loop = loop

        self.disk_lock = threading.Lock()

        store_path = config.extra_config.get("eloqstore.store_path")
        if not store_path:
            raise ValueError(
                "eloqstore.store_path is required in extra_config"
            )
        table_name = config.extra_config.get("eloqstore.table_name", "lmcache")
        num_threads = int(config.extra_config.get("eloqstore.num_threads", 4))

        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)

        extra = config.extra_config

        def _opt_int(key: str) -> int | None:
            v = extra.get(key)
            return int(v) if v is not None else None

        def _opt_bool(key: str) -> bool | None:
            v = extra.get(key)
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("true", "1", "yes")

        self.client = EloqStoreClient(
            store_path=store_path,
            table_name=table_name,
            num_threads=num_threads,
            data_page_size=_opt_int("eloqstore.data_page_size"),
            pages_per_file_shift=_opt_int("eloqstore.pages_per_file_shift"),
            data_append_mode=_opt_bool("eloqstore.data_append_mode"),
            overflow_pointers=_opt_int("eloqstore.overflow_pointers"),
            enable_compression=_opt_bool("eloqstore.enable_compression"),
            buffer_pool_size=_opt_int("eloqstore.buffer_pool_size"),
            manifest_limit=_opt_int("eloqstore.manifest_limit"),
            fd_limit=_opt_int("eloqstore.fd_limit"),
        )

        logger.info(
            "EloqStoreBackend initialized: path=%s table=%s threads=%d",
            store_path,
            table_name,
            num_threads,
        )

        self.eloqstore_worker = EloqStoreWorker(loop)
        self._read_executor = ThreadPoolExecutor(max_workers=8)

        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker: Optional["LMCacheWorker"] = None
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        self._restore_index()

    # ── index persistence ──────────────────────────────────────────

    def _restore_index(self) -> None:
        """Try to restore the in-memory key→metadata dict from EloqStore."""
        raw = self.client.get(_INDEX_KEY)
        if raw is None:
            logger.info("EloqStoreBackend: no persisted index found, starting fresh")
            return

        parsed = _deserialize_value(raw)
        if parsed is None:
            logger.warning("EloqStoreBackend: index header corrupt, starting fresh")
            return

        _, payload = parsed
        try:
            entries = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("EloqStoreBackend: index payload corrupt, starting fresh")
            return

        restored = 0
        total_size = 0
        with self.disk_lock:
            for key_str, meta_dict in entries.items():
                try:
                    key = CacheEngineKey.from_string(key_str)
                except (ValueError, KeyError):
                    logger.debug("Skipping unparseable index key: %s", key_str)
                    continue
                shape, dtype, fmt, size = _meta_from_dict(meta_dict)
                self.dict[key] = DiskCacheMetadata(
                    path="", size=size, shape=shape, dtype=dtype,
                    cached_positions=None, fmt=fmt, pin_count=0,
                )
                total_size += size
                restored += 1

        self.current_cache_size = float(total_size)
        self.usage = total_size
        self.stats_monitor.update_local_storage_usage(self.usage)
        logger.info(
            "EloqStoreBackend: restored %d entries (%.1f MB) from index",
            restored,
            total_size / (1024 * 1024),
        )

    def _save_index(self) -> None:
        """Persist the current in-memory dict to EloqStore."""
        entries: Dict[str, Dict[str, Any]] = {}
        with self.disk_lock:
            for key, meta in self.dict.items():
                if meta.shape is None or meta.dtype is None:
                    continue
                entries[key.to_string()] = _meta_to_dict(
                    meta.shape, meta.dtype, meta.fmt, meta.size,
                )

        payload = json.dumps(entries, separators=(",", ":")).encode("utf-8")
        # wrap in the same header format so _restore_index can read it
        wrapped = _serialize_value(
            {"version": 1, "count": len(entries)}, payload,
        )
        self.client.put(_INDEX_KEY, wrapped)
        logger.debug(
            "EloqStoreBackend: saved index with %d entries", len(entries),
        )

    # ── StoragePluginInterface ─────────────────────────────────────

    def set_worker(self, lmcache_worker: "LMCacheWorker") -> None:
        self.lmcache_worker = lmcache_worker
        if self.batched_msg_sender is None and self.config is not None \
                and self.metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=self.metadata,
                config=self.config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )

    def __str__(self) -> str:
        return "EloqStoreBackend"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self) -> None:
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.eloqstore_worker.exists_in_put_tasks(key)

    def pin(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.disk_lock.acquire()

        meta = self.dict.pop(key, None)
        if meta is None:
            if force:
                self.disk_lock.release()
            return False

        size = meta.size
        self.usage -= size

        self.client.delete(key.to_string())

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path="", size=size, shape=shape, dtype=dtype,
                    cached_positions=cached_positions, fmt=fmt, pin_count=0,
                )

        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[Future]:
        """Single-key put — delegates to batched path."""
        futs = self.batched_submit_put_task(
            [key], [memory_obj], on_complete_callback=on_complete_callback
        )
        return futs[0] if futs else None

    def _make_eviction_space(self, required: int) -> bool:
        """Evict entries until *required* bytes are available. Must hold disk_lock."""
        while self.current_cache_size + required > self.max_cache_size:
            evict_keys = self.cache_policy.get_evict_candidates(
                self.dict, num_candidates=1
            )
            if not evict_keys:
                logger.warning("No eviction candidates, space under pressure.")
                return False
            for ek in evict_keys:
                self.current_cache_size -= self.dict[ek].size
            self.batched_remove(evict_keys, force=False)
        return True

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        if not keys:
            return None

        # Filter out already-in-progress keys
        pending_keys: List[CacheEngineKey] = []
        pending_objs: List[MemoryObj] = []
        for k, o in zip(keys, objs, strict=True):
            if not self.exists_in_put_tasks(k):
                self.eloqstore_worker.insert_put_task(k)
                pending_keys.append(k)
                pending_objs.append(o)

        if not pending_keys:
            return None

        total_required = sum(o.get_physical_size() for o in pending_objs)
        with self.disk_lock:
            if not self._make_eviction_space(total_required):
                for k in pending_keys:
                    self.eloqstore_worker.remove_put_task(k)
                return None
            self.current_cache_size += total_required
            for k in pending_keys:
                self.cache_policy.update_on_put(k)

        for o in pending_objs:
            o.ref_count_up()

        fut = asyncio.run_coroutine_threadsafe(
            self.eloqstore_worker.submit_task(
                "put",
                self._async_batch_save_to_eloqstore,
                keys=pending_keys,
                objs=pending_objs,
                on_complete_callback=on_complete_callback,
            ),
            self.loop,
        )
        return [fut]

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        pending_keys: List[CacheEngineKey] = []
        pending_objs: List[MemoryObj] = []
        for k, o in zip(keys, objs, strict=True):
            if not self.exists_in_put_tasks(k):
                self.eloqstore_worker.insert_put_task(k)
                pending_keys.append(k)
                pending_objs.append(o)

        if not pending_keys:
            return

        total_required = sum(o.get_physical_size() for o in pending_objs)
        with self.disk_lock:
            if not self._make_eviction_space(total_required):
                for k in pending_keys:
                    self.eloqstore_worker.remove_put_task(k)
                return
            self.current_cache_size += total_required
            for k in pending_keys:
                self.cache_policy.update_on_put(k)

        for o in pending_objs:
            o.ref_count_up()

        await self.eloqstore_worker.submit_task(
            "put",
            self._async_batch_save_to_eloqstore,
            keys=pending_keys,
            objs=pending_objs,
            on_complete_callback=on_complete_callback,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _async_batch_save_to_eloqstore(
        self,
        keys: List[CacheEngineKey],
        objs: List[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Save multiple KV chunks in a single ``CEloqStore_PutBatch`` call."""
        batch_keys: List[str] = []
        batch_values: List[bytes] = []
        meta_list: List[tuple[CacheEngineKey, int, torch.Size, torch.dtype,
                               MemoryFormat, Optional[torch.Tensor]]] = []

        total_payload_bytes = 0
        for key, memory_obj in zip(keys, objs, strict=True):
            payload = bytes(memory_obj.byte_array)
            size = memory_obj.get_physical_size()
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            assert dtype is not None

            meta = _meta_to_dict(shape, dtype, fmt, size)
            value = _serialize_value(meta, payload)

            batch_keys.append(key.to_string())
            batch_values.append(value)
            cached_pos = memory_obj.metadata.cached_positions
            meta_list.append((key, size, shape, dtype, fmt, cached_pos))
            total_payload_bytes += len(payload)

        self.client.batch_put(batch_keys, batch_values)

        self.usage += total_payload_bytes
        self.stats_monitor.update_local_storage_usage(self.usage)

        for key, size, shape, dtype, fmt, cached_pos in meta_list:
            self.insert_key(key, size, shape, dtype, fmt,
                            cached_positions=cached_pos)

        for memory_obj in objs:
            memory_obj.ref_count_down()

        for key in keys:
            self.eloqstore_worker.remove_put_task(key)

        if on_complete_callback is not None:
            for key in keys:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(
                        "on_complete_callback failed for key %s: %s", key, e
                    )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.disk_lock:
            if key not in self.dict:
                return None

            disk_meta = self.dict[key]
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
            assert dtype is not None
            assert shape is not None

        memory_obj = self._load_from_eloqstore(
            key, dtype=dtype, shape=shape, fmt=fmt
        )

        if memory_obj is not None:
            with self.disk_lock:
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)

        return memory_obj

    def _load_from_eloqstore(
        self,
        key: CacheEngineKey,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        raw = self.client.get(key.to_string())
        if raw is None:
            return None

        parsed = _deserialize_value(raw)
        if parsed is None:
            logger.error("Corrupt data for key %s: invalid header", key)
            return None
        _, payload = parsed

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.error(
                "Memory allocation failed during EloqStore load for key %s.",
                key,
            )
            return None

        buffer = memoryview(memory_obj.byte_array)
        if buffer.format == "<B":
            buffer = buffer.cast("B")
        if len(payload) != len(buffer):
            logger.error(
                "Size mismatch for key %s: stored=%d, allocated=%d",
                key, len(payload), len(buffer),
            )
            return None

        buffer[:] = payload

        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        if not keys:
            return []
        # Parallelise individual gets through a thread pool.
        futs = [
            self._read_executor.submit(self.get_blocking, key)
            for key in keys
        ]
        return [f.result() for f in futs]

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    def get_allocator_backend(self) -> "LocalCPUBackend":
        return self.local_cpu_backend

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        # Stop worker first — wait for all in-flight puts to complete,
        # then persist the final index.
        self.eloqstore_worker.close()
        self._save_index()
        self._read_executor.shutdown(wait=True)
        self.client.close()
