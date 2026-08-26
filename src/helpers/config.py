from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "src" / ".env"
DEFAULT_STORY_PATH = PROJECT_ROOT / "data" / "examples" / "story.txt"
MODEL_CACHE_DIR = PROJECT_ROOT / ".hf-cache" / "hub"

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 1024
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None
TOP_K = None


def generation_kwargs() -> dict[str, Any]:
    settings = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }
    return {
        name: value
        for name, value in settings.items()
        if value is not None
    }
