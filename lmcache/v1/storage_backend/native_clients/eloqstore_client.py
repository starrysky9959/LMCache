# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class EloqStoreClient:
    """Thin synchronous wrapper around eloqstore.Client.

    Isolates all EloqStore-specific calls behind a small, deterministic API.
    Keys and values are ``bytes`` at this layer; callers are responsible for
    encoding/decoding to and from application-level types.
    """

    def __init__(
        self,
        store_path: str,
        table_name: str = "lmcache",
        num_threads: int = 4,
        data_page_size: int | None = None,
        pages_per_file_shift: int | None = None,
        data_append_mode: bool | None = None,
        overflow_pointers: int | None = None,
        enable_compression: bool | None = None,
        buffer_pool_size: int | None = None,
        manifest_limit: int | None = None,
        fd_limit: int | None = None,
    ) -> None:
        """Initialize and open an EloqStore instance.

        Args:
            store_path: Path to the EloqStore data directory.
            table_name: Logical table name.
            num_threads: Number of shard threads (io_uring rings).
            data_page_size: B+Tree page size in bytes (default 4KB, max 64KB).
            pages_per_file_shift: Data file = page_size << shift (default 18 = 1GB).
            data_append_mode: If True, use log-structured writes with aggregation.
            overflow_pointers: Max overflow pointers per page (default 16, max 128).
            enable_compression: Enable ZSTD compression.
            buffer_pool_size: Index page cache size per shard in bytes (default 32MB).
            manifest_limit: WAL file size limit in bytes (default 8MB).
            fd_limit: Max open files (default 10000).
        """
        # Third Party
        from eloqstore import Client, Options

        self._store_path = store_path
        self._table_name = table_name
        self._num_threads = num_threads

        self._options = Options(
            store_paths=[store_path],
            table_name=table_name,
            partition_id=0,
            num_threads=num_threads,
            data_page_size=data_page_size,
            pages_per_file_shift=pages_per_file_shift,
            data_append_mode=data_append_mode,
            overflow_pointers=overflow_pointers,
            enable_compression=enable_compression,
            buffer_pool_size=buffer_pool_size,
            manifest_limit=manifest_limit,
            fd_limit=fd_limit,
        )
        self._client = Client(self._options)
        logger.info(
            "EloqStore client opened: path=%s table=%s threads=%d "
            "page_size=%s append_mode=%s",
            store_path,
            table_name,
            num_threads,
            data_page_size,
            data_append_mode,
        )

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve the value for *key*, or ``None`` if not found.

        Args:
            key: Application-level key (will be UTF-8 encoded).

        Returns:
            Raw bytes stored under *key*, or ``None``.
        """
        return self._client.get(key.encode("utf-8"))

    def put(self, key: str, value: bytes) -> None:
        """Store *value* under *key*.

        Args:
            key: Application-level key (will be UTF-8 encoded).
            value: Raw bytes to store.
        """
        self._client.put(key.encode("utf-8"), value)

    def delete(self, key: str) -> bool:
        """Remove *key* from the store.

        Args:
            key: Application-level key (will be UTF-8 encoded).

        Returns:
            ``True`` if the key was deleted successfully.
        """
        self._client.delete(key.encode("utf-8"))
        return True

    def exists(self, key: str) -> bool:
        """Check whether *key* exists in the store.

        Args:
            key: Application-level key (will be UTF-8 encoded).

        Returns:
            ``True`` if the key exists.
        """
        return self._client.exists(key.encode("utf-8"))

    def batch_put(self, keys: List[str], values: List[bytes]) -> None:
        """Store multiple key-value pairs via a single ``CEloqStore_PutBatch`` call.

        Args:
            keys: Application-level keys.
            values: Corresponding raw byte values.
        """
        if not keys:
            return
        items = [(k.encode("utf-8"), v) for k, v in zip(keys, values, strict=True)]
        self._client.batch_put(items)

    def batch_get(self, keys: List[str]) -> List[Optional[bytes]]:
        """Retrieve values for multiple keys.

        Uses native ``CEloqStore_Get`` for each key; the C API does not
        expose a batch-get primitive.

        Args:
            keys: Application-level keys.

        Returns:
            List of values (``None`` for missing keys), same order as *keys*.
        """
        return [self.get(key) for key in keys]

    def batch_delete(self, keys: List[str]) -> int:
        """Remove multiple keys via a single ``CEloqStore_DeleteBatch`` call.

        Args:
            keys: Application-level keys.

        Returns:
            Number of keys deleted.
        """
        if not keys:
            return 0
        encoded = [k.encode("utf-8") for k in keys]
        self._client.batch_delete(encoded)
        return len(keys)

    def close(self) -> None:
        """Shut down the EloqStore client."""
        self._client.close()
        logger.info("EloqStore client closed: path=%s", self._store_path)
