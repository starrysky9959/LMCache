# EloqStore Backend

## Overview

`EloqStoreBackend` is a local persistent storage backend backed by
[EloqStore](https://pypi.org/project/eloqstore/), an embedded database
designed for efficient SSD storage.  It replaces the file-per-chunk I/O
pattern of `LocalDiskBackend` with a key-value store, allowing LMCache
to leverage EloqStore's internal LSM-tree and io_uring-based I/O engine.

## Integration Layer

```
GPU KV cache (hot tier)
  │
  ▼
LocalCPUBackend (CPU pinned memory, hot spill)
  │
  ▼
EloqStoreBackend (SSD persistent tier, via EloqStore embedded DB)
```

`EloqStoreBackend` implements `StoragePluginInterface` and is loaded
dynamically through LMCache's plugin mechanism (`storage_plugins`).
It sits at the same architectural layer as `LocalDiskBackend` — it is
a local persistent backend, not a remote connector.

## Data Model

### Storage key

```
{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype_str}
```

Example: `llama-7b@1@0@abcd@half`

### Value format

Each value is a self-describing binary blob:

```
┌──────────┬──────────┬──────────────────────┬──────────────────────────┐
│ 4B magic │ 4B len   │ JSON meta (~60B)     │ KV tensor raw bytes      │
│ "ELMC"   │ uint32 LE│ {"shape":[256,2,128],│ ~128KB (chunk=256,       │
│          │          │  "dtype":"float16",  │  single-layer) to        │
│          │          │  "fmt":1,"size":...} │  ~4MB (32 layers)        │
└──────────┴──────────┴──────────────────────┴──────────────────────────┘
```

The JSON metadata records `shape`, `dtype`, `fmt` (MemoryFormat), and `size`
(bytes).  The payload is the raw tensor bytes from `MemoryObj.byte_array` —
i.e. the complete KV cache chunk for all layers, without any transcoding.

### Global index

A well-known key `__lmcache_index__` stores the full `{key_str → meta_dict}`
mapping, wrapped in the same header format.  It is written on `close()` and
loaded on construction, enabling in-memory dictionary recovery across process
restarts.

## Design Decisions

### Plugin-style integration

Loaded via `storage_plugins: "eloqstore"` and `extra_config`, not hard-coded
in `CreateStorageBackends`.  This keeps the integration lightweight and
allows independent iteration.

### LMCache owns cache semantics

Eviction policy, pin/unpin state, capacity accounting, and usage tracking
remain in LMCache.  EloqStore is treated purely as a durable byte store.
This mirrors the ownership split in `LocalDiskBackend`.

### Synchronous SDK, async via thread pool

The EloqStore Python SDK (v0.1.0) wraps the C API via `ctypes` and exposes
synchronous methods (`put`, `get`, `exists`, `delete`).  To avoid blocking
the event loop, the backend routes these calls through
`AsyncPQThreadPoolExecutor` (same pattern as `LocalDiskBackend`).

### Metadata persistence

Each stored value embeds its own metadata (shape, dtype, fmt, size) in a
binary header.  A global index is persisted on clean shutdown and restored
on startup.  This supports cache reuse across process restarts without
requiring a full scan.

## Configuration

```yaml
storage_plugins: "eloqstore"
max_local_disk_size: 100  # GB, controls eviction threshold
extra_config:
  storage_plugin.eloqstore.module_path: lmcache.v1.storage_backend.eloqstore_backend
  storage_plugin.eloqstore.class_name: EloqStoreBackend
  eloqstore.store_path: "/data/lmcache-eloqstore"
  eloqstore.table_name: "lmcache"
  eloqstore.num_threads: 4
```

| Config key | Required | Default | Description |
|---|---|---|---|
| `eloqstore.store_path` | yes | — | Path to EloqStore data directory |
| `eloqstore.table_name` | no | `"lmcache"` | Logical table name |
| `eloqstore.num_threads` | no | `4` | EloqStore background threads |

## Files

| File | Purpose |
|---|---|
| `lmcache/v1/storage_backend/eloqstore_backend.py` | `EloqStoreBackend` + `EloqStoreWorker` |
| `lmcache/v1/storage_backend/native_clients/eloqstore_client.py` | Thin sync wrapper around `eloqstore.Client` |
| `tests/test_eloqstore_backend.py` | Smoke tests (round-trip, persistence) |

## Known Limitations

- **Single-process only**.  MP mode is not supported in this version.
- **No zero-copy GPU restore**.  Data is restored into CPU memory via
  `LocalCPUBackend` and then transferred to GPU by the upper layers.
- **Synchronous SDK calls**.  The EloqStore Python SDK does not yet expose
  native async/await methods.  Operations run in a thread pool via
  `asyncio.to_thread`.
- **Startup index loading is blocking**.  On construction the backend
  synchronously reads the global index from EloqStore.  For very large
  caches this may add noticeable startup latency.
