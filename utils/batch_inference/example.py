"""Example model representing a single batch inference request."""

from pydantic import BaseModel, field_validator

from utils.batch_inference.message import Message
from utils.batch_inference.types import ResponseSchema


class Example(BaseModel):
    """A single evaluation example within a dataset.

    Each example carries its own custom_id for result correlation,
    plus optional per-example response schema and system prompt.
    """

    model_config = {"arbitrary_types_allowed": True}

    custom_id: str
    messages: list[Message]
    response_schema: ResponseSchema | None = None
    system_prompt: str | None = None

    @field_validator("custom_id")
    @classmethod
    def custom_id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("custom_id must not be empty or whitespace-only")
        return v

    @field_validator("messages")
    @classmethod
    def messages_must_be_non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v
