from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class S3LoadedDocument:
    """Хранит идентификатор и локальную копию загруженного документа."""

    file_id: str
    filename: str
    char_count: int
    local_doc_id: str
