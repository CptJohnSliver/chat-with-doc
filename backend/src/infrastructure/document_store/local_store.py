"""On-disk document workspace keyed by local_doc_id (extracted text + optional source file)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from src.common.config.config import settings

_EXTRACTED_NAME = "extracted.txt"
_SOURCE_NAME = "source"
_SOFT_BOUNDARY_SEPARATORS = ("\n\n", "\n", ". ", " ")


class DocumentLocalStore:
    """Process-local registry of document workspaces under a temp root.

    Readers take a lease so `release_local_doc` defers `rmtree` until in-flight
    disk I/O (including work running inside `asyncio.to_thread`) finishes.
    """

    def __init__(self, *, root: Path | None = None) -> None:
        self._root = root
        self._workspaces: dict[str, Path] = {}
        self._lock = Lock()
        self._readers: dict[str, int] = {}
        self._pending_delete: set[str] = set()

    def max_upload_file_bytes(self) -> int:
        return int(settings.get("MAX_UPLOAD_FILE_BYTES", 50 * 1024 * 1024))

    def temp_root(self) -> Path:
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            return self._root
        default_root = Path(tempfile.gettempdir()) / "chat-with-documents"
        root = Path(os.environ.get("DOCUMENT_TEMP_DIR", str(default_root)))
        root.mkdir(parents=True, exist_ok=True)
        return root

    def allocate_workspace(self) -> tuple[str, Path]:
        local_doc_id = str(uuid.uuid4())
        workspace = self.temp_root() / local_doc_id
        workspace.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._workspaces[local_doc_id] = workspace
        return local_doc_id, workspace

    def cleanup_stale_workspaces(self, *, max_age_seconds: float) -> int:
        """Remove abandoned workspaces while preserving active work in this process."""
        cutoff = time.time() - max(0.0, max_age_seconds)
        with self._lock:
            active_paths = set(self._workspaces.values())
        removed = 0
        for workspace in self.temp_root().iterdir():
            if workspace in active_paths:
                continue
            try:
                if workspace.lstat().st_mtime > cutoff:
                    continue
                if workspace.is_symlink() or not workspace.is_dir():
                    workspace.unlink(missing_ok=True)
                else:
                    shutil.rmtree(workspace)
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return removed

    def get_workspace(self, local_doc_id: str) -> Path:
        with self._lock:
            if local_doc_id in self._pending_delete:
                raise KeyError(f"Workspace pending delete: {local_doc_id}")
            path = self._workspaces.get(local_doc_id)
        if path is None:
            raise KeyError(f"Unknown local_doc_id: {local_doc_id}")
        return path

    def extracted_text_path(self, local_doc_id: str) -> Path:
        return self.get_workspace(local_doc_id) / _EXTRACTED_NAME

    def source_file_path(self, local_doc_id: str, suffix: str) -> Path:
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        return self.get_workspace(local_doc_id) / f"{_SOURCE_NAME}{ext}"

    @contextmanager
    def _hold_readers(self, local_doc_id: str):
        """Keep the workspace alive for the duration of sync disk I/O."""
        with self._lock:
            if local_doc_id in self._pending_delete or local_doc_id not in self._workspaces:
                raise KeyError(f"Unknown local_doc_id: {local_doc_id}")
            self._readers[local_doc_id] = self._readers.get(local_doc_id, 0) + 1
        try:
            yield
        finally:
            self._release_reader(local_doc_id)

    def _release_reader(self, local_doc_id: str) -> None:
        workspace: Path | None = None
        with self._lock:
            count = self._readers.get(local_doc_id, 0) - 1
            if count > 0:
                self._readers[local_doc_id] = count
            else:
                self._readers.pop(local_doc_id, None)
                if local_doc_id in self._pending_delete:
                    self._pending_delete.discard(local_doc_id)
                    workspace = self._workspaces.pop(local_doc_id, None)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)

    def read_extracted_text(self, local_doc_id: str) -> str:
        with self._hold_readers(local_doc_id):
            path = self._workspaces[local_doc_id] / _EXTRACTED_NAME
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8")

    def read_extracted_snippet(self, local_doc_id: str, max_chars: int) -> str:
        with self._hold_readers(local_doc_id):
            path = self._workspaces[local_doc_id] / _EXTRACTED_NAME
            with path.open(encoding="utf-8") as handle:
                return handle.read(max_chars).strip()

    def extracted_byte_size(self, local_doc_id: str) -> int:
        with self._hold_readers(local_doc_id):
            path = self._workspaces[local_doc_id] / _EXTRACTED_NAME
            return path.stat().st_size if path.exists() else 0

    @staticmethod
    def _soft_cut_index(text: str, lookback: int) -> int | None:
        """Prefer cutting near the end of a full window on a natural boundary."""
        if lookback <= 0 or len(text) <= lookback:
            return None
        region_start = len(text) - lookback
        region = text[region_start:]
        for sep in _SOFT_BOUNDARY_SEPARATORS:
            idx = region.rfind(sep)
            if idx != -1:
                return region_start + idx + len(sep)
        return None

    def iter_extracted_text_windows(
        self,
        local_doc_id: str,
        *,
        max_chars: int,
        soft_boundary_lookback: int = 500,
    ) -> Iterator[str]:
        """Yield consecutive text windows from disk without loading the whole file."""
        with self._hold_readers(local_doc_id):
            path = self._workspaces[local_doc_id] / _EXTRACTED_NAME
            max_chars = max(1, max_chars)
            lookback = min(max(0, soft_boundary_lookback), max_chars // 4)
            leftover = ""

            with path.open(encoding="utf-8") as handle:
                while True:
                    need = max_chars - len(leftover)
                    data = handle.read(need) if need > 0 else ""
                    if not data and not leftover:
                        break

                    buf = leftover + data
                    leftover = ""

                    if not data or len(buf) < max_chars:
                        stripped = buf.strip()
                        if stripped:
                            yield stripped
                        break

                    cut = self._soft_cut_index(buf, lookback) or max_chars
                    cut = max(1, min(cut, len(buf)))
                    piece = buf[:cut]
                    leftover = buf[cut:]
                    stripped = piece.strip()
                    if stripped:
                        yield stripped

    def release_local_doc(self, local_doc_id: str) -> None:
        """Drop the workspace. If readers are active, defer rmtree until they finish."""
        workspace: Path | None = None
        with self._lock:
            if local_doc_id not in self._workspaces:
                return
            if self._readers.get(local_doc_id, 0) > 0:
                self._pending_delete.add(local_doc_id)
                return
            self._pending_delete.discard(local_doc_id)
            workspace = self._workspaces.pop(local_doc_id, None)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)

    def release_attached_local_docs(self, local_doc_ids: list[str | None]) -> None:
        for doc_id in local_doc_ids:
            if doc_id:
                self.release_local_doc(doc_id)


document_local_store = DocumentLocalStore()
