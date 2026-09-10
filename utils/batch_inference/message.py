"""Message model for batch inference requests."""

from pydantic import BaseModel, field_validator

from utils.batch_inference.types import Role


class Message(BaseModel):
    """A single conversation turn. System prompts are handled separately."""

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def content_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v
