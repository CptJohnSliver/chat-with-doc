from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileMetadata:
    file_id: str
    bucket_name: str
    original_filename: str
    storage_key: str
    size_bytes: int
    content_type: str | None
    created_by: str | None = None
