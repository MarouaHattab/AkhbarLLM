from pydantic import BaseModel, ConfigDict


class LoadTestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    response_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    average_input_tokens: float
    average_output_tokens: float
