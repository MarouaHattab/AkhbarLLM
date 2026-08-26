from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "src" / ".env"
DEFAULT_STORY_PATH = PROJECT_ROOT / "data" / "examples" / "story.txt"
MODEL_CACHE_DIR = PROJECT_ROOT / ".hf-cache" / "hub"

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MAX_NEW_TOKENS = 1024
QWEN_DO_SAMPLE = False
QWEN_TEMPERATURE = None
QWEN_TOP_P = None
QWEN_TOP_K = None
QWEN_CUDA_DEVICE = "cuda"
QWEN_CPU_DEVICE = "cpu"
QWEN_CUDA_DTYPE = "float16"
QWEN_CPU_DTYPE = "float32"
QWEN_DEVICE_MAP = "auto"

GEMINI_MODEL_ID = "gemini-3.1-flash-lite"
GEMINI_MAX_OUTPUT_TOKENS = 1024
GEMINI_TEMPERATURE = 0.0

# Backward-compatible name used by existing training code.
BASE_MODEL_ID = QWEN_MODEL_ID


def qwen_generation_kwargs() -> dict[str, Any]:
    settings = {
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "do_sample": QWEN_DO_SAMPLE,
        "temperature": QWEN_TEMPERATURE,
        "top_p": QWEN_TOP_P,
        "top_k": QWEN_TOP_K,
    }
    return {
        name: value
        for name, value in settings.items()
        if value is not None
    }


# Backward-compatible helper used by the existing Qwen evaluator.
generation_kwargs = qwen_generation_kwargs
