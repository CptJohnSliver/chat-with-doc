import uuid
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from src.common.schemas.models import HistoryMessage


class DocumentRequest(BaseModel):
    message_id: UUID = Field(default_factory=uuid.uuid4)
    history_messages: list[HistoryMessage] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_last_user_message(self) -> Self:
        if not self.history_messages:
            raise ValueError("history_messages cannot be empty")
        last_message = self.history_messages[-1]
        if last_message.role != "user":
            raise ValueError("Last message in history_messages must have role='user'")
        if len(last_message.fragments) != 1:
            raise ValueError("Last user message must have exactly one fragment")
        fragment = last_message.fragments[0]
        if fragment.fragment_type != "request":
            raise ValueError("The fragment of the last user message must have type 'request'")
        return self
