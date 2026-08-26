from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    task: Literal["extraction", "translation"]
    raw_response: str
    json_valid: bool
    schema_valid: bool
    validation_error: str | None = None
