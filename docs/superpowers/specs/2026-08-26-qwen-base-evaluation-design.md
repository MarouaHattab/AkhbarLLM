# Qwen Base-Model Evaluation Design

## Goal

Add a reusable command-line evaluation pipeline for `Qwen/Qwen2.5-1.5B-Instruct` that runs the existing news-details extraction and Arabic-to-English translation tasks against `data/examples/story.txt` before fine-tuning.

The evaluator must authenticate with the Hugging Face token in `src/.env`, load the base model once, choose a safe device and dtype automatically, generate deterministic responses, and report both raw output and structured-output validation results.

## Scope

This feature evaluates the local Hugging Face base model only. It does not fine-tune a model, call OpenAI or Gemini, perform knowledge distillation, compare model scores across a dataset, or execute paid API requests.

## Architecture

### `src/helpers/config.py`

Keep non-secret runtime configuration in Python constants:

- `BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"`
- `MAX_NEW_TOKENS = 1024`
- `DO_SAMPLE = False`
- `TEMPERATURE = None`
- `TOP_P = None`
- `TOP_K = None`
- the project-relative default story path

The existing fixed `DEVICE` and `TORCH_DTYPE` constants will be removed because the runtime must discover them automatically.

### `src/helpers/huggingface.py`

Load `src/.env` with `python-dotenv`, require a non-empty `HF_TOKEN`, authenticate with `huggingface_hub.login` without adding the token to Git credentials, and verify the token with `huggingface_hub.whoami`. The token must never be logged or included in exceptions.

### `src/models/qwen.py`

Own the Hugging Face model runtime:

- choose CUDA plus `torch.float16` when `torch.cuda.is_available()` is true;
- otherwise choose CPU plus `torch.float32`;
- load `AutoModelForCausalLM` and `AutoTokenizer` using the configured model ID and authenticated token;
- use `device_map="auto"` for CUDA and explicit CPU placement for the fallback;
- set the tokenizer pad token to its EOS token when no pad token exists;
- call `model.eval()` after loading;
- render messages with `tokenizer.apply_chat_template`;
- tokenize with padding and move inputs to the model input device;
- generate inside `torch.inference_mode()`;
- slice prompt tokens from every generated sequence;
- decode only the newly generated tokens.

The optional sampling values remain in configuration but are omitted from `model.generate` when they are `None`, preventing irrelevant sampling warnings while `do_sample=False`.

### `src/models/evaluation.py`

Define an `EvaluationResult` model with:

- task name;
- raw model response;
- whether the response is valid JSON;
- whether the decoded JSON matches the task's Pydantic schema;
- a safe validation error string when validation fails.

Invalid JSON or schema output is a valid base-model evaluation result, not a process-level failure.

### `src/utils/story_loader.py`

Read a UTF-8 story file. Accept both plain text and the repository's current legacy format, `story = """..."""`. Reject missing and empty files with clear errors.

### `src/controllers/evaluation.py`

Provide extraction and translation evaluation functions. Each function:

1. builds messages with the existing task module;
2. generates one response through the shared Qwen runtime;
3. validates the response against `NewsDetails` or `TranslatedStory`;
4. returns an `EvaluationResult` without printing.

This keeps orchestration separate from the model runtime and terminal presentation.

### `src/evaluate.py`

Provide the command:

```powershell
python -m src.evaluate --task extraction
python -m src.evaluate --task translation
python -m src.evaluate --task both
```

`--task` accepts `extraction`, `translation`, or `both` and defaults to `both`. The CLI also accepts a story path and source/target languages, defaulting to the repository example, Arabic, and English. It authenticates once, loads the model once, runs the selected tasks sequentially, and prints labeled raw responses plus validation status.

## Data Flow

```text
src/.env -> Hugging Face authentication -> Qwen runtime
data/examples/story.txt -> story loader -> task message builder
task messages -> chat template -> tokenizer -> model.generate
generated tokens -> decoded response -> JSON/Pydantic validation -> CLI output
```

## Error Handling

The command exits unsuccessfully with a concise message when:

- `HF_TOKEN` is missing or empty;
- Hugging Face rejects the token;
- the story file is missing or empty;
- the model or tokenizer cannot be downloaded or loaded;
- generation itself fails.

The command continues normally when model output is not valid JSON or violates the requested schema because exposing that limitation is the purpose of pre-fine-tuning evaluation.

## Runtime Expectations

The current virtual environment has CPU-only PyTorch. Therefore, the first verified run will use CPU with `float32` unless the environment is later changed to a CUDA-enabled PyTorch build. Loading a 1.5-billion-parameter model and generating up to 1024 tokens twice can be slow and requires a multi-gigabyte model download and sufficient system memory.

## Testing

Automated unit tests will cover:

- all generation settings being sourced from configuration;
- CUDA/`float16` and CPU/`float32` runtime selection;
- missing Hugging Face token handling without secret disclosure;
- pad-token fallback;
- chat-template application, input placement, prompt-token slicing, and decoding;
- plain-text and legacy triple-quoted story loading;
- extraction and translation schema validation;
- invalid model JSON being returned as an evaluation result;
- CLI task selection and one-time runtime loading.

Network/model tests will be separated from unit tests. Final verification will first run the complete unit-test suite, then authenticate with the configured token, load the real Qwen model, and execute both tasks on the example story.

## Security

Only API-key names belong in `src/.env.example`. Real values remain in ignored `src/.env`. The evaluator never prints tokens and never passes them as command-line arguments.
