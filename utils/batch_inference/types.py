"""Type aliases for batch inference configuration models."""

from typing import Any, Literal

from pydantic import BaseModel

# Schema can be a Pydantic model class, raw dict, or JSON string
ResponseSchema = type[BaseModel] | dict[str, Any] | str

Role = Literal["user", "assistant"]
