# Qwen Base-Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic pre-fine-tuning evaluator for Qwen extraction and translation using the repository example story.

**Architecture:** Configuration and Hugging Face authentication remain in `helpers`; Qwen loading and generation remain in `models`; task orchestration remains in `controllers`; file normalization remains in `utils`; and `src.evaluate` is the only CLI entry point. The model is loaded once and shared between selected tasks, while invalid structured output is recorded as an evaluation result instead of crashing the command.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Hugging Face Hub, python-dotenv, Pydantic, unittest

---

### Task 1: Runtime Configuration and Example Loading

**Files:**
- Modify: `src/helpers/config.py`
- Create: `src/utils/story_loader.py`
- Modify: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `tests/test_story_loader.py`

- [ ] **Step 1: Write failing configuration tests**

```python
from unittest import TestCase

from src.helpers import config


class ConfigTests(TestCase):
    def test_generation_settings_match_base_evaluation(self):
        self.assertEqual(config.BASE_MODEL_ID, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(config.MAX_NEW_TOKENS, 1024)
        self.assertFalse(config.DO_SAMPLE)
        self.assertIsNone(config.TEMPERATURE)
        self.assertIsNone(config.TOP_P)
        self.assertIsNone(config.TOP_K)

    def test_generation_kwargs_omit_disabled_sampling_values(self):
        self.assertEqual(
            config.generation_kwargs(),
            {"max_new_tokens": 1024, "do_sample": False},
        )
```

- [ ] **Step 2: Run configuration tests and verify they fail because the settings/function are missing**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_config -v`

Expected: FAIL for missing generation constants or `generation_kwargs`.

- [ ] **Step 3: Implement configuration constants and generation kwargs**

```python
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "src" / ".env"
DEFAULT_STORY_PATH = PROJECT_ROOT / "data" / "examples" / "story.txt"
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
    return {name: value for name, value in settings.items() if value is not None}
```

- [ ] **Step 4: Write failing story-loader tests**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.utils.story_loader import load_story


class StoryLoaderTests(TestCase):
    def test_loads_plain_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "story.txt"
            path.write_text("  خبر عربي  ", encoding="utf-8")
            self.assertEqual(load_story(path), "خبر عربي")

    def test_unwraps_legacy_python_assignment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "story.txt"
            path.write_text('story = """\nخبر عربي\n"""', encoding="utf-8")
            self.assertEqual(load_story(path), "خبر عربي")

    def test_rejects_empty_story(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "story.txt"
            path.write_text("   ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_story(path)
```

- [ ] **Step 5: Run story-loader tests and verify the missing-module failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_story_loader -v`

Expected: ERROR because `src.utils.story_loader` does not exist.

- [ ] **Step 6: Implement the UTF-8 story loader**

```python
from pathlib import Path


def load_story(path: str | Path) -> str:
    story_path = Path(path)
    text = story_path.read_text(encoding="utf-8").strip()
    prefix = 'story = """'
    if text.startswith(prefix) and text.endswith('"""'):
        text = text[len(prefix):-3].strip()
    if not text:
        raise ValueError(f"Story file is empty: {story_path}")
    return text
```

- [ ] **Step 7: Run both test modules and commit**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_config tests.test_story_loader -v`

Expected: all Task 1 tests PASS.

```powershell
git add src/helpers/config.py src/utils tests/test_config.py tests/test_story_loader.py
git commit -m "feat: add evaluation configuration and story loader"
```

### Task 2: Hugging Face Authentication

**Files:**
- Create: `src/helpers/huggingface.py`
- Modify: `src/helpers/__init__.py`
- Create: `tests/test_huggingface.py`

- [ ] **Step 1: Write failing authentication tests with injected Hub functions**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.helpers.huggingface import authenticate_huggingface


class HuggingFaceTests(TestCase):
    def test_requires_hf_token(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                authenticate_huggingface(env_path=env_path, environ={})

    def test_logs_in_and_verifies_the_env_token(self):
        calls = []
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=secret-token\n", encoding="utf-8")

            token = authenticate_huggingface(
                env_path=env_path,
                environ={},
                login_fn=lambda **kwargs: calls.append(("login", kwargs)),
                whoami_fn=lambda **kwargs: calls.append(("whoami", kwargs)) or {"name": "user"},
            )

        self.assertEqual(token, "secret-token")
        self.assertEqual(calls[0][1]["token"], "secret-token")
        self.assertFalse(calls[0][1]["add_to_git_credential"])
        self.assertEqual(calls[1][1]["token"], "secret-token")

    def test_authentication_error_does_not_expose_token(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=secret-token\n", encoding="utf-8")
            with self.assertRaises(RuntimeError) as context:
                authenticate_huggingface(
                    env_path=env_path,
                    environ={},
                    login_fn=lambda **_: (_ for _ in ()).throw(ValueError("bad")),
                )
        self.assertNotIn("secret-token", str(context.exception))
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_huggingface -v`

Expected: ERROR because `src.helpers.huggingface` does not exist.

- [ ] **Step 3: Implement `.env` authentication with safe dependency injection**

```python
import os
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from huggingface_hub import login, whoami

from src.helpers.config import ENV_PATH


def authenticate_huggingface(
    env_path: str | Path = ENV_PATH,
    environ: MutableMapping[str, str] | None = None,
    login_fn: Callable[..., Any] = login,
    whoami_fn: Callable[..., Any] = whoami,
) -> str:
    target_environ = os.environ if environ is None else environ
    file_values = dotenv_values(env_path)
    token = (target_environ.get("HF_TOKEN") or file_values.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(f"HF_TOKEN is missing or empty in {Path(env_path)}")
    try:
        login_fn(token=token, add_to_git_credential=False, skip_if_logged_in=False)
        whoami_fn(token=token)
    except Exception as exc:
        raise RuntimeError("Hugging Face authentication failed.") from exc
    return token
```

- [ ] **Step 4: Run the authentication tests and commit**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_huggingface -v`

Expected: all Task 2 tests PASS and no token appears in output.

```powershell
git add src/helpers/huggingface.py src/helpers/__init__.py tests/test_huggingface.py
git commit -m "feat: authenticate Qwen evaluation with Hugging Face"
```

### Task 3: Qwen Runtime and Deterministic Generation

**Files:**
- Create: `src/models/qwen.py`
- Modify: `src/models/__init__.py`
- Create: `tests/test_qwen.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Write failing runtime-selection tests**

```python
from unittest import TestCase

import torch

from src.models.qwen import select_runtime_target


class RuntimeTargetTests(TestCase):
    def test_selects_cuda_float16(self):
        target = select_runtime_target(cuda_available=lambda: True)
        self.assertEqual(target.device, "cuda")
        self.assertIs(target.dtype, torch.float16)
        self.assertEqual(target.device_map, "auto")

    def test_selects_cpu_float32(self):
        target = select_runtime_target(cuda_available=lambda: False)
        self.assertEqual(target.device, "cpu")
        self.assertIs(target.dtype, torch.float32)
        self.assertIsNone(target.device_map)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_qwen.RuntimeTargetTests -v`

Expected: ERROR because `src.models.qwen` does not exist.

- [ ] **Step 3: Implement `RuntimeTarget` and automatic selection**

```python
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeTarget:
    device: str
    dtype: torch.dtype
    device_map: str | None


def select_runtime_target(
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
) -> RuntimeTarget:
    if cuda_available():
        return RuntimeTarget("cuda", torch.float16, "auto")
    return RuntimeTarget("cpu", torch.float32, None)
```

- [ ] **Step 4: Write failing loader and generation tests using lightweight fakes**

```python
from typing import Any
from unittest import TestCase

import torch

from src.models.qwen import QwenRuntime, RuntimeTarget


class FakeBatch(dict):
    def __init__(self, input_ids: list[list[int]]):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeModel:
    def __init__(self, device="cpu", generated_ids=None):
        self.device = device
        self.generated_ids = generated_ids or [[10, 11, 20, 21]]
        self.load_kwargs: dict[str, Any] = {}
        self.generate_kwargs: dict[str, Any] = {}
        self.eval_called = False
        self.moved_to = None

    def record_load(self, kwargs):
        self.load_kwargs = kwargs
        return self

    def to(self, device):
        self.moved_to = device
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.generated_ids


class FakeTokenizer:
    def __init__(
        self,
        pad_token_id=0,
        eos_token="<eos>",
        input_ids=None,
        decoded="response",
    ):
        self.pad_token_id = pad_token_id
        self.pad_token = None
        self.eos_token = eos_token
        self.input_ids = input_ids or [[10, 11]]
        self.decoded = decoded
        self.add_generation_prompt = False
        self.decoded_ids = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.add_generation_prompt = add_generation_prompt
        return "rendered prompt"

    def __call__(self, texts, return_tensors, padding):
        return FakeBatch(self.input_ids)

    def batch_decode(self, token_ids, skip_special_tokens):
        self.decoded_ids = token_ids
        return [self.decoded]


class QwenRuntimeTests(TestCase):
    def test_load_sets_pad_token_and_cpu_device(self):
        model = FakeModel(device="cpu")
        tokenizer = FakeTokenizer(pad_token_id=None, eos_token="<eos>")
        runtime = QwenRuntime.load(
            token="token",
            target=RuntimeTarget("cpu", torch.float32, None),
            model_loader=lambda *_, **kwargs: model.record_load(kwargs),
            tokenizer_loader=lambda *_, **__: tokenizer,
        )
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertTrue(model.eval_called)
        self.assertEqual(model.moved_to, "cpu")
        self.assertIs(runtime.model, model)

    def test_generate_applies_chat_template_and_decodes_only_new_tokens(self):
        model = FakeModel(device="cpu", generated_ids=[[10, 11, 20, 21]])
        tokenizer = FakeTokenizer(input_ids=[[10, 11]], decoded="response")
        runtime = QwenRuntime(model=model, tokenizer=tokenizer)
        response = runtime.generate([{"role": "user", "content": "story"}])
        self.assertEqual(response, "response")
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.decoded_ids, [[20, 21]])
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 1024)
        self.assertFalse(model.generate_kwargs["do_sample"])
```

- [ ] **Step 5: Run the generation tests and verify they fail for missing behavior**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_qwen.QwenRuntimeTests -v`

Expected: FAIL because `QwenRuntime` does not yet provide `load` and `generate`.

- [ ] **Step 6: Implement model loading and generation**

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.config import BASE_MODEL_ID, generation_kwargs


@dataclass(frozen=True)
class RuntimeTarget:
    device: str
    dtype: torch.dtype
    device_map: str | None


def select_runtime_target(
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
) -> RuntimeTarget:
    if cuda_available():
        return RuntimeTarget("cuda", torch.float16, "auto")
    return RuntimeTarget("cpu", torch.float32, None)


class QwenRuntime:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        token: str,
        target: RuntimeTarget | None = None,
        model_loader: Callable[..., Any] = AutoModelForCausalLM.from_pretrained,
        tokenizer_loader: Callable[..., Any] = AutoTokenizer.from_pretrained,
    ) -> "QwenRuntime":
        selected = target or select_runtime_target()
        model_kwargs: dict[str, Any] = {
            "token": token,
            "torch_dtype": selected.dtype,
        }
        if selected.device_map is not None:
            model_kwargs["device_map"] = selected.device_map
        model = model_loader(BASE_MODEL_ID, **model_kwargs)
        if selected.device_map is None:
            model.to(selected.device)
        model.eval()
        tokenizer = tokenizer_loader(BASE_MODEL_ID, token=token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model, tokenizer)

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs())
        new_ids = [
            output[len(prompt):]
            for prompt, output in zip(inputs.input_ids, output_ids)
        ]
        return self.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
        )[0].strip()
```

- [ ] **Step 7: Declare evaluation runtime dependencies and refresh the lock file**

Add this exact dependency group, then refresh the lock file:

```toml
[dependency-groups]
evaluation = [
    "accelerate>=1.2.1,<2",
    "huggingface-hub>=0.36.0,<2",
    "torch>=2.5.1",
    "transformers==4.48.3",
]
```

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv lock
uv sync --group evaluation
```

Expected: dependency resolution and environment synchronization complete successfully.

- [ ] **Step 8: Run Qwen unit tests and commit**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_qwen -v`

Expected: all Task 3 tests PASS without downloading a model.

```powershell
git add pyproject.toml uv.lock src/models tests/test_qwen.py
git commit -m "feat: add reusable Qwen model runtime"
```

### Task 4: Evaluation Results and Controllers

**Files:**
- Create: `src/models/evaluation.py`
- Modify: `src/models/__init__.py`
- Create: `src/controllers/evaluation.py`
- Modify: `src/controllers/__init__.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

```python
from unittest import TestCase

from src.controllers.evaluation import evaluate_extraction, evaluate_translation


class FakeRuntime:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.response


class EvaluationTests(TestCase):
    def test_extraction_reports_valid_schema_output(self):
        response = '{"story_title":"عنوان صالح","story_keywords":["مال"],"story_summary":["ملخص"],"story_category":"economy","story_entities":[{"entity_value":"فوربس","entity_type":"organization"}]}'
        result = evaluate_extraction(FakeRuntime(response), "خبر عربي طويل")
        self.assertTrue(result.json_valid)
        self.assertTrue(result.schema_valid)
        self.assertIsNone(result.validation_error)

    def test_invalid_json_is_an_evaluation_result(self):
        result = evaluate_extraction(FakeRuntime("not json"), "خبر عربي طويل")
        self.assertFalse(result.json_valid)
        self.assertFalse(result.schema_valid)
        self.assertIsNotNone(result.validation_error)

    def test_translation_uses_requested_languages(self):
        runtime = FakeRuntime('{"translated_title":"Valid title","translated_content":"Valid content"}')
        result = evaluate_translation(runtime, "خبر عربي", "French", "Arabic")
        self.assertTrue(result.schema_valid)
        self.assertIn("French", runtime.messages[1]["content"])
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_evaluation -v`

Expected: ERROR because the evaluation controller and result model do not exist.

- [ ] **Step 3: Implement `EvaluationResult` and controller validation**

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["extraction", "translation"]
    raw_response: str
    json_valid: bool
    schema_valid: bool
    validation_error: str | None = None
```

```python
import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from src.models.evaluation import EvaluationResult
from src.models.news import NewsDetails
from src.models.qwen import QwenRuntime
from src.models.translation import TranslatedStory
from src.tasks import build_extraction_messages, build_translation_messages


def _validate_response(
    task: Literal["extraction", "translation"],
    raw_response: str,
    schema: type[BaseModel],
) -> EvaluationResult:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return EvaluationResult(
            task=task,
            raw_response=raw_response,
            json_valid=False,
            schema_valid=False,
            validation_error=str(exc),
        )
    try:
        schema.model_validate(payload)
    except ValidationError as exc:
        return EvaluationResult(
            task=task,
            raw_response=raw_response,
            json_valid=True,
            schema_valid=False,
            validation_error=str(exc),
        )
    return EvaluationResult(
        task=task,
        raw_response=raw_response,
        json_valid=True,
        schema_valid=True,
    )


def evaluate_extraction(runtime: QwenRuntime, story: str) -> EvaluationResult:
    raw_response = runtime.generate(build_extraction_messages(story))
    return _validate_response("extraction", raw_response, NewsDetails)


def evaluate_translation(
    runtime: QwenRuntime,
    story: str,
    target_language: str = "English",
    source_language: str = "Arabic",
) -> EvaluationResult:
    messages = build_translation_messages(
        story,
        target_language=target_language,
        source_language=source_language,
    )
    raw_response = runtime.generate(messages)
    return _validate_response("translation", raw_response, TranslatedStory)
```

- [ ] **Step 4: Run controller tests and commit**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_evaluation -v`

Expected: all Task 4 tests PASS.

```powershell
git add src/controllers src/models tests/test_evaluation.py
git commit -m "feat: evaluate extraction and translation output"
```

### Task 5: CLI and Real Base-Model Evaluation

**Files:**
- Create: `src/evaluate.py`
- Create: `tests/test_evaluate_cli.py`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing CLI orchestration tests**

```python
from pathlib import Path
from unittest import TestCase

from src.evaluate import run_evaluations
from src.models.evaluation import EvaluationResult


class FakeRuntime:
    pass


def valid_result(task: str) -> EvaluationResult:
    return EvaluationResult(
        task=task,
        raw_response="{}",
        json_valid=True,
        schema_valid=True,
    )


class EvaluateCliTests(TestCase):
    def test_both_tasks_authenticate_and_load_runtime_once(self):
        calls = []
        results = run_evaluations(
            task="both",
            story_path="story.txt",
            source_language="Arabic",
            target_language="English",
            authenticate=lambda: calls.append("authenticate") or "token",
            runtime_loader=lambda token: calls.append(("load", token)) or FakeRuntime(),
            story_loader=lambda _: "خبر عربي",
            extraction_evaluator=lambda *_: calls.append("extraction") or valid_result("extraction"),
            translation_evaluator=lambda *_: calls.append("translation") or valid_result("translation"),
        )
        self.assertEqual(calls.count("authenticate"), 1)
        self.assertEqual(calls.count(("load", "token")), 1)
        self.assertEqual([result.task for result in results], ["extraction", "translation"])

    def test_extraction_selection_skips_translation(self):
        calls = []
        results = run_evaluations(
            task="extraction",
            story_path=Path("story.txt"),
            source_language="Arabic",
            target_language="English",
            authenticate=lambda: "token",
            runtime_loader=lambda _: FakeRuntime(),
            story_loader=lambda _: "خبر عربي",
            extraction_evaluator=lambda *_: calls.append("extraction") or valid_result("extraction"),
            translation_evaluator=lambda *_args, **_kwargs: calls.append("translation") or valid_result("translation"),
        )
        self.assertEqual([result.task for result in results], ["extraction"])
        self.assertEqual(calls, ["extraction"])
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.test_evaluate_cli -v`

Expected: ERROR because `src.evaluate` does not exist.

- [ ] **Step 3: Implement CLI parsing, one-time setup, task selection, and formatting**

Implement the complete CLI module:

```python
import argparse
from collections.abc import Sequence
from pathlib import Path

from src.controllers.evaluation import evaluate_extraction, evaluate_translation
from src.helpers.config import DEFAULT_STORY_PATH
from src.helpers.huggingface import authenticate_huggingface
from src.models.evaluation import EvaluationResult
from src.models.qwen import QwenRuntime
from src.utils.story_loader import load_story


def run_evaluations(
    task: str,
    story_path: str | Path,
    source_language: str,
    target_language: str,
    authenticate=authenticate_huggingface,
    runtime_loader=QwenRuntime.load,
    story_loader=load_story,
    extraction_evaluator=evaluate_extraction,
    translation_evaluator=evaluate_translation,
) -> list[EvaluationResult]:
    story = story_loader(story_path)
    token = authenticate()
    runtime = runtime_loader(token)
    results = []
    if task in {"extraction", "both"}:
        results.append(extraction_evaluator(runtime, story))
    if task in {"translation", "both"}:
        results.append(
            translation_evaluator(
                runtime,
                story,
                target_language=target_language,
                source_language=source_language,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the base Qwen model.")
    parser.add_argument(
        "--task",
        choices=("extraction", "translation", "both"),
        default="both",
    )
    parser.add_argument("--story", type=Path, default=DEFAULT_STORY_PATH)
    parser.add_argument("--source-language", default="Arabic")
    parser.add_argument("--target-language", default="English")
    return parser


def print_result(result: EvaluationResult) -> None:
    print(f"\n=== {result.task.upper()} ===")
    print(result.raw_response)
    print(f"JSON valid: {result.json_valid}")
    print(f"Schema valid: {result.schema_valid}")
    if result.validation_error:
        print(f"Validation error: {result.validation_error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run_evaluations(
            task=args.task,
            story_path=args.story,
            source_language=args.source_language,
            target_language=args.target_language,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    for result in results:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Document usage and ignore local model caches**

Add `.hf-cache/` to `.gitignore` and document:

```powershell
uv sync --group evaluation
python -m src.evaluate --task extraction
python -m src.evaluate --task translation
python -m src.evaluate --task both
```

- [ ] **Step 5: Run the entire unit-test suite**

Run: `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`

Expected: every test PASS without model download or network access.

- [ ] **Step 6: Verify Hugging Face authentication without exposing the token**

Run a Python command that calls `authenticate_huggingface()` and prints only `Hugging Face authentication succeeded`.

Expected: success message and no token value.

- [ ] **Step 7: Run extraction and translation with the real base model**

Run:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.hf-cache'
python -m src.evaluate --task both
```

Expected: the model loads using automatic CPU/`float32` fallback in the current environment; both labeled raw responses and their JSON/schema validation statuses are printed. A schema-invalid response is acceptable evaluation evidence.

- [ ] **Step 8: Run final static and repository checks**

Run:

```powershell
.venv\\Scripts\\python.exe -m compileall -q src tests
git diff --check
git status --short
```

Expected: compilation and diff checks succeed; status lists only intentional feature changes.

- [ ] **Step 9: Commit the completed evaluator**

```powershell
git add .gitignore README.md src/evaluate.py tests/test_evaluate_cli.py
git commit -m "feat: add Qwen base model evaluation CLI"
```
