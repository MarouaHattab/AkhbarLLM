from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["extraction", "translation"]
    raw_response: str
    json_valid: bool
    schema_valid: bool
    validation_error: str | None = None
