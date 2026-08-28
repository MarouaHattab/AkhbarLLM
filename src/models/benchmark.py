from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    provider: str
    model_id: str
    sample_count: int
    total_seconds: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def average_latency_seconds(self) -> float:
        return self.total_seconds / self.sample_count

    @property
    def requests_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.sample_count / self.total_seconds

    @property
    def tokens_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.total_tokens / self.total_seconds
