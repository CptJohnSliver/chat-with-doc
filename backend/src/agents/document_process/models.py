"""Domain models for document_process."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AttachedDocument(BaseModel):
    """Хранит исходный идентификатор, текст и происхождение вложения."""

    file_id: str
    filename: str
    text: str = ""
    file_index: int
    char_count: int = 0
    local_doc_id: str | None = None
    #: Отличает текущее вложение от документа из истории.
    from_current_message: bool = True

    @property
    def display_name(self) -> str:
        return self.filename or f"Документ {self.file_index + 1}"


class Chunk(BaseModel):
    """Связывает выбранную часть текста с исходным документом."""

    file_id: str
    doc_name: str
    text: str
    segment_index: int | None = Field(default=None, ge=0)
    selection_reason: Literal["search", "neighbor", "full", "fallback"] = "full"
    #: Происхождение документа, из которого взят кусок. См. AttachedDocument.
    from_current_message: bool = True


class RelevantDocs(BaseModel):
    document: list[Chunk] = Field(default_factory=list)


def _merge_snapshot(
    model_type: type[BaseModel],
    left: BaseModel | None,
    right: BaseModel | None,
) -> BaseModel:
    if left is right and left is not None:
        return left
    if left is None and right is None:
        return model_type()
    if left is not None and right is not None:
        try:
            if left.model_dump() == right.model_dump():
                return left
        except Exception:
            pass
    return right or left or model_type()


def merge_relevant_docs(left: RelevantDocs | None, right: RelevantDocs | None) -> RelevantDocs:
    return _merge_snapshot(RelevantDocs, left, right)  # type: ignore[return-value]
