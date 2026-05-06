# EloqStore Backend Implementation Plan

## Summary

This document proposes the recommended LMCache integration path for
EloqStore in non-MP mode.

The core decision is:

- Treat EloqStore as an in-process local storage backend.
- Integrate it at the `lmcache.v1.storage_backend` layer.
- Do not model it as a `RemoteBackend`.
- Defer MP/L2-adapter work until the single-process path is stable.

This plan is optimized for the current development environment, where
single-process validation is practical and MP mode is difficult to test
reliably under WSL.

## Problem Statement

EloqStore is an embedded database. It is designed to run in the same
process as its caller and to manage local persistent storage directly.

LMCache currently has two broad storage integration shapes:

- Local in-process storage backends, such as `LocalCPUBackend` and
  `LocalDiskBackend`.
- Remote-style backends, wrapped by `RemoteBackend`, which assume an
  external storage system and a connector boundary.

Using EloqStore through `RemoteBackend` is a poor semantic fit:

- It treats an embedded store as if it were an external service.
- It adds connector lifecycle and remote-oriented control flow that are
  unnecessary for an in-process engine.
- It pushes the design toward serializer/connector/reconnect semantics
  that make sense for Redis or S3, but not for a local embedded store.

The better fit is to treat EloqStore as the storage engine behind a
local persistent backend, replacing the file-per-chunk behavior of
`LocalDiskBackend`.

## Goals

- Add a production-oriented EloqStore-backed storage backend for
  non-MP LMCache.
- Preserve the current LMCache storage hierarchy:
  GPU cache -> `LocalCPUBackend` -> persistent local storage.
- Reuse LMCache's existing cache policy, pin/unpin, and lookup flow.
- Make the first implementation testable in single-process mode on WSL.
- Keep the design reusable for a later MP integration.

## Non-Goals

- Do not make MP mode the primary implementation target.
- Do not expose EloqStore first as a `remote_url` backend.
- Do not migrate LMCache eviction policy into EloqStore.
- Do not optimize for multi-node or network sharing in the first phase.
- Do not attempt zero-copy direct GPU restore in the first phase.

## Recommended Architecture

### Integration Layer

Implement a new backend under `lmcache/v1/storage_backend/`:

- `EloqStoreBackend`, implementing `StoragePluginInterface`.

This puts EloqStore in the same architectural tier as local persistent
storage, rather than remote storage.

### Data Flow

The target hierarchy is:

- GPU KV cache remains the hot tier.
- `LocalCPUBackend` remains the memory allocator and hot spill tier.
- `EloqStoreBackend` becomes the persistent local tier.

Operationally:

1. LMCache writes KV chunks to `LocalCPUBackend`.
2. Evicted or asynchronously persisted chunks are written into
   `EloqStoreBackend`.
3. Reads from `EloqStoreBackend` allocate a `MemoryObj` through
   `LocalCPUBackend` and restore bytes into that object.
4. The existing `StorageManager` write-back flow keeps hot data in
   `LocalCPUBackend` after retrieval.

### Ownership Split

LMCache should continue owning cache semantics:

- in-memory key index
- eviction policy
- pin / unpin state
- usage accounting
- ongoing put-task tracking

EloqStore should own durable byte storage:

- persistent object storage for chunk payloads
- optional persistence for metadata blobs if needed
- local SSD layout and write/read mechanics

This keeps the integration aligned with existing LMCache backend design.

## Why Not `RemoteBackend`

`RemoteBackend` is a compatibility layer for remote or remote-like
storage connectors. It assumes:

- local CPU buffers as a staging area
- connector-driven async I/O
- remote serializer / deserializer flow
- remote health and reconnection semantics

Those assumptions are appropriate for Redis, RESP, S3, and similar
backends. They are not the cleanest boundary for an embedded store.

Even if EloqStore can be made to work through that interface, the design
would still be suboptimal because:

- the abstraction boundary is wrong
- control flow becomes harder to reason about
- future local-storage optimizations become awkward

## Phased Implementation Plan

### Phase 0: Scope and Validation Harness

Goal: establish the minimum viable development path before backend work.

Tasks:

- Confirm the EloqStore Python/C++ binding shape needed by LMCache.
- Decide whether the first version links via an internal pybind module or
  imports a separately built package.
- Define a small single-process smoke test target that does not depend on
  vLLM MP mode.

Deliverables:

- import/build notes
- a repeatable local smoke test command

### Phase 1: Native Client Wrapper

Goal: isolate EloqStore-specific storage calls behind a small client API.

Add:

- `lmcache/v1/storage_backend/native_clients/eloqstore_client.py`
  or equivalent wrapper module

Responsibilities:

- initialize/open the EloqStore instance
- map LMCache operations to simple storage methods
- provide:
  - `get(key) -> bytes | None`
  - `set(key, value: bytes) -> None`
  - `delete(key) -> bool`
  - `exists(key) -> bool`
  - optional batched variants for get/set/delete/exists
- own EloqStore-specific config parsing
- own close/shutdown behavior

Constraints:

- keep the API synchronous first
- avoid introducing LMCache cache semantics into this layer
- keep all key mapping deterministic and explicit

Why this phase matters:

- it prevents EloqStore details from leaking throughout the backend
- it gives a reusable primitive for a later MP implementation

### Phase 2: `EloqStoreBackend`

Goal: add the non-MP LMCache backend that uses the native client.

Add:

- `lmcache/v1/storage_backend/eloqstore_backend.py`

Implement `StoragePluginInterface` methods:

- `contains`
- `exists_in_put_tasks`
- `batched_submit_put_task`
- `async_batched_submit_put_task`
- `get_blocking`
- `batched_get_blocking`
- `pin`
- `unpin`
- `remove`
- `close`

Recommended behavior:

- follow `LocalDiskBackend` structure where practical
- keep an in-memory mutable mapping of
  `CacheEngineKey -> metadata/index record`
- use LMCache cache policy for eviction ordering
- use `AsyncPQThreadPoolExecutor` for async put/delete work
- allocate restore buffers through `LocalCPUBackend`

Metadata stored in LMCache memory should include:

- logical key
- byte size
- dtype
- shape
- memory format
- optional cached positions
- pin state or pin-compatible metadata
- optional EloqStore object identifier if different from the logical key

### Phase 3: Configuration Surface

Goal: make the backend selectable through normal LMCache config.

Recommended configuration path:

- use `storage_plugins`
- avoid `remote_url`

Example shape:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 8
storage_plugins: "eloqstore"
extra_config:
  storage_plugin.eloqstore.module_path: lmcache.v1.storage_backend.eloqstore_backend
  storage_plugin.eloqstore.class_name: EloqStoreBackend
  eloqstore.store_paths: "/data/lmcache-eloqstore"
  eloqstore.table_name: "lmcache"
  eloqstore.branch: "main"
  eloqstore.num_workers: 4
  eloqstore.threads: 1
  eloqstore.max_capacity_gb: 100
```

Implementation tasks:

- define a stable config key namespace under `extra_config`
- support sensible defaults
- validate config early and fail loudly

Optional follow-up:

- later add first-class config fields to `LMCacheEngineConfig` if the
  backend graduates from plugin-style integration to built-in status

### Phase 4: Eviction and Capacity Accounting

Goal: preserve LMCache backend behavior parity with `LocalDiskBackend`.

Requirements:

- maintain `current_cache_size`
- enforce `max_capacity_gb`
- evict according to configured cache policy
- never evict pinned entries
- keep usage metrics updated

Recommended approach:

- keep capacity accounting in LMCache, not in EloqStore
- use synchronous metadata mutation guarded by a backend lock
- use async physical deletes if needed, but commit logical removal
  consistently from LMCache's point of view

Open implementation choice:

- Whether delete should be synchronous in the first version for simpler
  correctness, then made async later if profiling justifies it.

Recommendation:

- start with correctness-first delete behavior, even if simpler than the
  most aggressive async design

### Phase 5: Batched Restore Optimization

Goal: avoid a slow one-key-at-a-time restore path.

First version:

- it is acceptable for `batched_get_blocking` to call single-key restore
  in a loop if correctness is easier

Preferred version:

- add native batched get support in the EloqStore client
- allocate all destination `MemoryObj`s first
- perform a grouped restore from EloqStore into those buffers

This phase is important for performance, but it should not block the
first usable implementation.

### Phase 6: Documentation and Tests

Goal: make the backend maintainable and easy to validate.

Tests to add:

- unit tests for config parsing
- unit tests for metadata/index mutation
- put/get round-trip test
- exists/remove test
- pin/unpin and eviction exclusion test
- capacity-driven eviction test
- batched get/put behavior test
- smoke test in CPU-only or non-CUDA environment if possible

Docs to add after implementation:

- user-facing storage backend page under
  `docs/source/kv_cache/storage_backends/`
- developer-facing note in
  `docs/source/developer_guide/extending_lmcache/storage_plugins.rst`

## File-Level Plan

### New Files

- `lmcache/v1/storage_backend/eloqstore_backend.py`
- `lmcache/v1/storage_backend/native_clients/eloqstore_client.py`
- tests for backend behavior and config
- user-facing backend doc

### Existing Files Likely Touched

- `lmcache/v1/storage_backend/__init__.py`
  only if we decide to register EloqStore as a built-in backend instead of
  a dynamically loaded plugin
- docs indexes if a published user doc is added

## Key Design Decisions

### Decision 1: Built-in backend vs plugin-style backend

Recommendation:

- implement as a `StoragePluginInterface` backend first

Reason:

- lowest-risk path
- minimal disturbance to LMCache core backend construction
- easiest way to iterate while the feature is still experimental

Possible later transition:

- promote to built-in backend once the config shape and runtime behavior
  are stable

### Decision 2: Reuse LMCache eviction policy

Recommendation:

- yes, reuse LMCache eviction policy

Reason:

- aligns with `LocalDiskBackend`
- avoids splitting cache semantics across two systems
- keeps behavior predictable for LMCache users

### Decision 3: Persist metadata only in LMCache or also in EloqStore

Recommendation:

- first version keeps authoritative runtime metadata in LMCache memory
- store enough payload structure in EloqStore to restore bytes correctly
- do not block the first version on crash recovery or restart reindexing

Reason:

- runtime correctness is the immediate need
- restart-time recovery can be added later with a clear migration path

### Decision 4: MP support timing

Recommendation:

- explicitly defer MP support

Reason:

- current environment makes MP validation unreliable
- non-MP correctness should be proven first
- the native client from Phase 1 can be reused later by an MP adapter

## Risks and Mitigations

### Risk: WSL-specific I/O behavior differs from Linux production

Mitigation:

- make correctness the first milestone
- keep benchmarks separate from architectural validation
- avoid overfitting early design to WSL-specific performance numbers

### Risk: Async put/delete introduces index inconsistency

Mitigation:

- define strict ordering rules for when metadata becomes visible
- track in-flight put tasks explicitly
- keep the first implementation conservative

### Risk: Batched get path underperforms

Mitigation:

- treat batched restore as an explicit later phase
- ensure the native client API leaves room for batched operations

### Risk: The plugin path feels too temporary

Mitigation:

- keep config naming stable from the start
- keep module boundaries production-quality
- promote to built-in only after behavior stabilizes

## Acceptance Criteria

The first implementation is successful when all of the following are true:

- EloqStore runs in-process with LMCache in non-MP mode.
- KV chunks can be asynchronously persisted and synchronously restored.
- `StorageManager` can use the backend without `RemoteBackend`.
- Eviction respects LMCache policy and pinning.
- The backend is testable on the current WSL environment.
- No MP server is required for basic validation.

## Follow-Up Work After Phase 1-6

Once the non-MP backend is stable, the next design step should be:

1. evaluate whether the native client API is sufficient for MP reuse
2. decide whether MP should use:
   - a dedicated `L2AdapterInterface` implementation, or
   - a native-connector-based adapter bridge
3. add restart/recovery indexing if persistent cache reuse across process
   restarts becomes important

## Final Recommendation

Implement EloqStore first as a non-MP in-process storage backend using
`StoragePluginInterface`, with LMCache retaining cache semantics and
EloqStore providing durable local chunk storage.

This is the cleanest architectural fit, the easiest path to validate
under WSL, and the best foundation for any later MP-specific work.
