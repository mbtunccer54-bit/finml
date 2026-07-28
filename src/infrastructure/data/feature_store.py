"""A local feature store: Parquet for values, SQLite for metadata.

The problem it solves is narrow and real. A FinBERT sentiment score costs a
transformer forward pass per document; recomputing it every time an obligor is
scored is wasteful and, worse, non-reproducible if the model or the news window
shifts underneath. The store computes such features once, records *how* they
were produced, and serves them back on subsequent runs.

Two access patterns are supported:

* **Versioned datasets** — write a whole frame under a name and version, read it
  back for lineage or replay.
* **Keyed feature groups** — upsert rows keyed by ``(entity_id, as_of_date)``
  and look up a subset, reporting which keys are missing so the caller computes
  only those.

Metadata records whether a value came from a real model or a degraded fallback,
so a fallback score is never silently mistaken for model output.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from domain.exceptions import FeatureStoreError
from infrastructure.logging import get_logger

__all__ = ["FeatureSetRecord", "FeatureStore"]

_log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_sets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    version       TEXT    NOT NULL,
    path          TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    n_rows        INTEGER NOT NULL,
    n_columns     INTEGER NOT NULL,
    columns_json  TEXT    NOT NULL,
    key_columns   TEXT    NOT NULL DEFAULT '',
    checksum      TEXT    NOT NULL,
    producer      TEXT    NOT NULL DEFAULT '',
    is_fallback   INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT    NOT NULL DEFAULT '{}',
    UNIQUE (name, version)
);
CREATE INDEX IF NOT EXISTS idx_feature_sets_name ON feature_sets (name, created_at DESC);
"""


@dataclass(frozen=True, slots=True)
class FeatureSetRecord:
    """Metadata for one stored feature set.

    Attributes:
        name: Feature set name.
        version: Version label.
        path: Parquet file location.
        created_at: Write timestamp.
        n_rows: Row count.
        n_columns: Column count.
        columns: Column names.
        key_columns: Columns forming the natural key, if any.
        checksum: Content hash, used to detect silent corruption.
        producer: What produced the values, for example ``finbert``.
        is_fallback: Whether a degraded fallback produced the values.
        metadata: Free-form extra detail.
    """

    name: str
    version: str
    path: str
    created_at: datetime
    n_rows: int
    n_columns: int
    columns: tuple[str, ...] = field(default_factory=tuple)
    key_columns: tuple[str, ...] = field(default_factory=tuple)
    checksum: str = ""
    producer: str = ""
    is_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """Parquet-backed feature storage with a SQLite metadata catalogue.

    Attributes:
        base_dir: Root directory holding the Parquet files and the catalogue.
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialise the store, creating the directory and catalogue.

        Args:
            base_dir: Root directory for the store.

        Raises:
            FeatureStoreError: If the directory or catalogue cannot be created.
        """
        self.base_dir = Path(base_dir)
        self._db_path = self.base_dir / "catalogue.sqlite"
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise FeatureStoreError(
                "Could not initialise the feature store",
                base_dir=str(self.base_dir),
                reason=str(exc),
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transactional connection to the catalogue.

        Yields:
            An open connection; committed on success, rolled back on failure.

        Raises:
            FeatureStoreError: If the catalogue rejects the operation.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise FeatureStoreError(
                "Feature store catalogue operation failed", reason=str(exc)
            ) from exc
        finally:
            conn.close()

    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        """Compute a content hash for a frame.

        Args:
            df: The frame to hash.

        Returns:
            A truncated SHA-256 digest of the row hashes.
        """
        hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
        return hashlib.sha256(hashed.tobytes()).hexdigest()[:32]

    def _parquet_path(self, name: str, version: str) -> Path:
        """Build the Parquet path for a feature set version.

        Args:
            name: Feature set name.
            version: Version label.

        Returns:
            The file path.
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self.base_dir / safe_name / f"{version}.parquet"

    # -- Versioned datasets --------------------------------------------------
    def write(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        version: str | None = None,
        key_columns: Sequence[str] = (),
        producer: str = "",
        is_fallback: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSetRecord:
        """Write a feature set and catalogue it.

        Args:
            name: Feature set name.
            df: The frame to store.
            version: Version label; a UTC timestamp is generated when omitted.
            key_columns: Columns forming the natural key.
            producer: What produced the values, for example ``finbert``.
            is_fallback: Whether a degraded fallback produced the values.
            metadata: Free-form extra detail.

        Returns:
            The catalogue record for the write.

        Raises:
            FeatureStoreError: If the frame is empty, a key column is absent, or
                the write fails.
        """
        if df.empty:
            raise FeatureStoreError("Refusing to store an empty feature set", name=name)
        missing_keys = [c for c in key_columns if c not in df.columns]
        if missing_keys:
            raise FeatureStoreError(
                "Key columns absent from the frame", name=name, missing=missing_keys
            )

        resolved_version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self._parquet_path(name, resolved_version)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        except (OSError, ValueError, ImportError) as exc:
            raise FeatureStoreError(
                "Could not write the feature set parquet",
                name=name,
                version=resolved_version,
                path=str(path),
                reason=str(exc),
            ) from exc

        record = FeatureSetRecord(
            name=name,
            version=resolved_version,
            path=str(path),
            created_at=datetime.now(UTC),
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=tuple(str(c) for c in df.columns),
            key_columns=tuple(key_columns),
            checksum=self._checksum(df),
            producer=producer,
            is_fallback=is_fallback,
            metadata=metadata or {},
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feature_sets
                    (name, version, path, created_at, n_rows, n_columns, columns_json,
                     key_columns, checksum, producer, is_fallback, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (name, version) DO UPDATE SET
                    path = excluded.path,
                    created_at = excluded.created_at,
                    n_rows = excluded.n_rows,
                    n_columns = excluded.n_columns,
                    columns_json = excluded.columns_json,
                    key_columns = excluded.key_columns,
                    checksum = excluded.checksum,
                    producer = excluded.producer,
                    is_fallback = excluded.is_fallback,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.name,
                    record.version,
                    record.path,
                    record.created_at.isoformat(),
                    record.n_rows,
                    record.n_columns,
                    json.dumps(list(record.columns)),
                    ",".join(record.key_columns),
                    record.checksum,
                    record.producer,
                    int(record.is_fallback),
                    json.dumps(record.metadata),
                ),
            )

        _log.info(
            "feature_store.written",
            name=name,
            version=resolved_version,
            n_rows=record.n_rows,
            n_columns=record.n_columns,
            producer=producer,
            is_fallback=is_fallback,
        )
        return record

    def read(self, name: str, version: str | None = None) -> pd.DataFrame:
        """Read a stored feature set.

        Args:
            name: Feature set name.
            version: Version label; the most recent is used when omitted.

        Returns:
            The stored frame.

        Raises:
            FeatureStoreError: If the version does not exist or cannot be read.
        """
        record = self.get_record(name, version)
        if record is None:
            raise FeatureStoreError(
                "Feature set not found", name=name, version=version or "<latest>"
            )
        try:
            return pd.read_parquet(record.path)
        except (OSError, ValueError, ImportError) as exc:
            raise FeatureStoreError(
                "Could not read the feature set parquet",
                name=name,
                version=record.version,
                path=record.path,
                reason=str(exc),
            ) from exc

    def get_record(self, name: str, version: str | None = None) -> FeatureSetRecord | None:
        """Look up catalogue metadata.

        Args:
            name: Feature set name.
            version: Version label; the most recent is used when omitted.

        Returns:
            The record, or ``None`` when absent.
        """
        with self._connect() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM feature_sets WHERE name = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM feature_sets WHERE name = ? AND version = ?",
                    (name, version),
                ).fetchone()
        return self._to_record(row) if row is not None else None

    @staticmethod
    def _to_record(row: sqlite3.Row) -> FeatureSetRecord:
        """Convert a catalogue row into a record.

        Args:
            row: The SQLite row.

        Returns:
            The corresponding record.
        """
        return FeatureSetRecord(
            name=row["name"],
            version=row["version"],
            path=row["path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            n_rows=int(row["n_rows"]),
            n_columns=int(row["n_columns"]),
            columns=tuple(json.loads(row["columns_json"])),
            key_columns=tuple(c for c in row["key_columns"].split(",") if c),
            checksum=row["checksum"],
            producer=row["producer"],
            is_fallback=bool(row["is_fallback"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def exists(self, name: str, version: str | None = None) -> bool:
        """Check whether a feature set version is catalogued.

        Args:
            name: Feature set name.
            version: Version label; the most recent is used when omitted.

        Returns:
            ``True`` when a matching record exists.
        """
        return self.get_record(name, version) is not None

    def list_sets(self) -> list[FeatureSetRecord]:
        """List every catalogued feature set version.

        Returns:
            All records, most recent first.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feature_sets ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def delete(self, name: str, version: str) -> bool:
        """Remove a feature set version and its Parquet file.

        Args:
            name: Feature set name.
            version: Version label.

        Returns:
            ``True`` when a record was removed.
        """
        record = self.get_record(name, version)
        if record is None:
            return False
        try:
            Path(record.path).unlink(missing_ok=True)
        except OSError as exc:
            # Losing the catalogue entry while the file lingers is the lesser
            # evil, but it must be visible.
            _log.warning(
                "feature_store.orphaned_file", path=record.path, reason=str(exc)
            )
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM feature_sets WHERE name = ? AND version = ?", (name, version)
            )
        _log.info("feature_store.deleted", name=name, version=version)
        return True

    # -- Keyed feature groups ------------------------------------------------
    def upsert_keyed(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        producer: str = "",
        is_fallback: bool = False,
    ) -> FeatureSetRecord:
        """Merge rows into a keyed feature group.

        Existing keys are replaced by the incoming values, which is what makes a
        recomputation with a newer NLP model take effect without a full rebuild.

        Args:
            name: Feature group name.
            df: Rows to merge, including the key columns.
            key_columns: Columns forming the natural key.
            producer: What produced the values.
            is_fallback: Whether a degraded fallback produced the values.

        Returns:
            The catalogue record for the merged group.

        Raises:
            FeatureStoreError: If key columns are missing or the write fails.
        """
        keys = list(key_columns)
        missing = [c for c in keys if c not in df.columns]
        if missing:
            raise FeatureStoreError(
                "Key columns absent from the frame", name=name, missing=missing
            )

        existing_record = self.get_record(name)
        if existing_record is not None:
            existing = self.read(name)
            merged = pd.concat([existing, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
        else:
            merged = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

        return self.write(
            name,
            merged,
            version="current",
            key_columns=keys,
            producer=producer,
            is_fallback=is_fallback,
            metadata={"updated_at": datetime.now(UTC).isoformat()},
        )

    def fetch_keyed(
        self, name: str, keys: pd.DataFrame, *, key_columns: Sequence[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Look up rows for a set of keys.

        Args:
            name: Feature group name.
            keys: Frame carrying the keys to look up.
            key_columns: Columns forming the natural key.

        Returns:
            ``(found, missing)`` — the matched rows, and the keys with no stored
            value so the caller computes only those.
        """
        key_list = list(key_columns)
        wanted = keys[key_list].drop_duplicates()

        if not self.exists(name):
            _log.debug("feature_store.miss_all", name=name, n_keys=len(wanted))
            return pd.DataFrame(columns=key_list), wanted

        stored = self.read(name)
        found = wanted.merge(stored, on=key_list, how="inner")
        missing = wanted.merge(stored[key_list], on=key_list, how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"][key_list].reset_index(drop=True)

        _log.debug(
            "feature_store.lookup",
            name=name,
            n_requested=len(wanted),
            n_found=len(found),
            n_missing=len(missing),
            hit_rate=round(len(found) / max(len(wanted), 1), 4),
        )
        return found.reset_index(drop=True), missing
