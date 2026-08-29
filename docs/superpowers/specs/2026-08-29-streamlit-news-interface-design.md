# Streamlit News Interface Design

## Objective

Build a production-quality Streamlit interface for the existing Arabic news
fine-tuning project. The application exposes information extraction and
translation, streams genuine decoder output as it is generated, loads the
fine-tuned model once per process, and preserves every existing synchronous
evaluation, benchmark, vLLM, Locust, training, and dataset workflow.

## Existing System

The project fine-tunes a rank-64 LoRA adapter over
`Qwen/Qwen2.5-1.5B-Instruct`. The adapter is available locally at
`outputs/models/news-finetune` and publicly on the Hub as
`marouaHattab/ArabLLM-news`.

The existing runtime boundaries are:

- `src.models.factory.load_language_model()` selects a provider.
- `FineTunedQwenModel` loads the Qwen base model, attaches the LoRA adapter,
  switches the model to evaluation mode, and installs Chinese-token
  suppression.
- `VLLMModel` sends OpenAI-compatible chat-completion requests to the served
  `news-lora` adapter.
- `build_extraction_messages()` owns the extraction prompt and embeds the
  `NewsDetails` JSON schema.
- `build_translation_messages()` owns source/target normalization, the
  translation prompt, and the `TranslatedStory` JSON schema.
- Existing controllers call the synchronous `generate()` method and validate
  completed JSON for evaluation reports.

No runtime currently exposes streaming generation. The application therefore
needs an additive streaming contract rather than UI-specific generation code.

## Chosen Architecture

The application uses a shared streaming-runtime abstraction:

```text
Streamlit entry point
    -> cached NewsStreamingController
        -> existing task prompt builder
            -> StreamingLanguageModel.stream(messages)
                -> FineTunedQwenModel / QwenModel (default)
                -> VLLMModel (optional remote backend)
        -> completed-response schema validation
```

This approach keeps Streamlit ignorant of tokenizers, CUDA, LoRA attachment,
OpenAI-compatible payloads, and task schemas. It also makes local development
and later remote vLLM deployment use the same UI and controller.

### Alternatives Rejected

Embedding `TextIteratorStreamer` directly in the Streamlit page would be a
smaller initial patch, but it would duplicate model generation behavior in the
UI and bypass the runtime abstraction.

Making vLLM mandatory would provide continuous batching, but it would prevent
the requested local application from running unless a separate Linux/WSL
server were already available. vLLM remains an optional provider selected by
configuration.

Artificial character-by-character playback after synchronous generation is
not acceptable because it does not improve time to first output and hides the
model's real latency.

## Component Boundaries

### Streaming model protocol

`src/models/language_model.py` will define a runtime-checkable
`StreamingLanguageModel` protocol that extends the existing `LanguageModel`
contract with:

```python
def stream(self, messages: list[ChatMessage]) -> Iterator[str]: ...
```

The current `LanguageModel` and `TokenizedLanguageModel` protocols remain
compatible. Existing callers continue using `generate()` without modification.

### Transformers streaming

`QwenModel.stream()` will:

1. Apply the tokenizer's existing chat template with a generation prompt.
2. Tokenize and move inputs to the model device exactly as synchronous
   generation does.
3. Build generation settings with `qwen_generation_kwargs()` and preserve any
   configured logits processors.
4. Use Transformers `TextIteratorStreamer` with prompt and special tokens
   omitted.
5. Run `model.generate()` in a worker thread while the calling thread yields
   real decoded chunks immediately.
6. Capture worker exceptions, terminate the streamer, join the worker, and
   re-raise a useful `RuntimeError` after already-produced output is drained.
7. Raise an error if generation completes without non-whitespace output.

`FineTunedQwenModel` inherits this implementation, so it automatically streams
with the attached LoRA adapter and Chinese-token suppressor.

The existing synchronous `QwenModel.generate()` implementation remains in
place to minimize regression risk.

### vLLM streaming

`VLLMModel.stream()` will call the existing OpenAI client with `stream=True`
and yield non-empty `choices[0].delta.content` fragments. It retains the
configured model ID, maximum-token limit, temperature, timeout behavior, and
post-generation CJK safety check. The existing non-streaming `generate()`
method remains unchanged.

### Fine-tuned adapter source

Local execution continues to prefer `outputs/models/news-finetune`, preserving
the current behavior. `FINETUNED_ADAPTER_SOURCE` will optionally select a Hub
adapter such as `marouaHattab/ArabLLM-news` for clean deployments where the
ignored local weights directory is absent. Adapter-source resolution and
validation remain in `src/models/finetuned_qwen.py`, not in Streamlit.

### Task streaming controller

A focused controller in `src/controllers/streaming.py` will:

- Accept only `extraction` or `translation`.
- Invoke the existing prompt builders without copying prompt text or schemas.
- Serialize access to the cached runtime with a lock so two Streamlit sessions
  cannot concurrently mutate or over-allocate a single model instance.
- Expose `stream_task(...)`, which yields runtime chunks without buffering the
  complete response first.
- Expose `validate_task_response(task, raw_response)`, which validates the
  caller's final accumulated response against `NewsDetails` or
  `TranslatedStory`.
- Represent validation failures separately from transport/model failures so
  the UI can preserve and display malformed raw output for diagnosis.

The controller has no dependency on Streamlit.

## Streamlit Application

A thin root `app.py` starts the application implemented in
`src/ui/streamlit_app.py`.

`@st.cache_resource(show_spinner=False)` will cache one controller and its
model runtime for the lifetime of the Streamlit process. Changing task,
language, uploaded file, or widget state therefore reruns the page without
reloading Qwen or the LoRA adapter.

The runtime provider is selected by `NEWS_MODEL_PROVIDER`:

- `finetuned` is the default and loads the fine-tuned Transformers model.
- `vllm` uses the existing OpenAI-compatible vLLM client.

The UI never contains access tokens or inference credentials.

### Input behavior

Users can paste a story or upload one UTF-8 `.txt` or `.md` file. A pasted
story takes precedence when both sources contain text, and the interface states
that rule next to the inputs. UTF-8 with or without a byte-order mark is
accepted. Empty files, invalid encoding, and empty submissions produce inline
validation messages without starting model inference.

Extraction requires only the story. Translation additionally provides source
and target language fields, initialized to Arabic and English and passed to the
existing normalization logic.

### Streaming behavior

Submitting a valid request creates an assistant-style result panel immediately.
Before the first chunk, it displays a `Generating` status. The first real chunk
changes the state to `Receiving model output`, and a single placeholder is
updated with the accumulated JSON text as each subsequent chunk arrives.

The UI does not attempt to parse incomplete JSON. After the stream ends, it
validates the complete response:

- Valid extraction JSON becomes an editorial result view containing title,
  category, keywords, summary points, and an entity table.
- Valid translation JSON becomes translated title and content.
- Invalid JSON or schema output remains visible as raw text alongside the
  precise validation error.

The raw response remains available after successful validation for copying and
debugging.

### Visual direction

The interface uses a restrained Swiss editorial system appropriate for a news
workbench: white or neutral background, dark text, one deliberate red accent,
hairline grid rules, left-aligned Latin text, and correct right-to-left
rendering for Arabic input and output. The memorable move is a live editorial
proof panel whose structured fields snap to the same visible column rhythm as
the input surface.

The page uses a two-column input/result composition on wide screens and stacks
the same regions on narrow screens. It contains no fabricated metrics,
decorative icons, gradient effects, or themed replacements for standard
actions.

## Error Model

The UI distinguishes:

- Input errors: empty story, empty file, unsupported extension, or invalid
  UTF-8.
- Model-loading errors: missing local adapter, inaccessible Hub adapter,
  missing Hugging Face credentials when required, or insufficient memory.
- Generation errors: CUDA allocation failures, worker-thread failures, empty
  output, vLLM connection failures, or malformed streaming responses.
- Output errors: invalid JSON or valid JSON that violates the selected task's
  Pydantic schema.

User-facing messages explain the next action without exposing API keys or full
internal tracebacks. Full exception details are logged server-side. Any partial
model text already received remains visible when a later error occurs.

## Testing Strategy

Implementation follows red-green-refactor TDD.

Fast automated tests will cover:

- Streaming protocol conformance.
- Transformers streaming yields real chunks, preserves generation settings and
  logits processors, and propagates worker failures.
- vLLM streaming extracts delta content and handles empty or malformed streams.
- The task controller reuses the correct extraction/translation prompt builder,
  serializes model access, and validates the correct schema.
- UTF-8 upload decoding and input precedence.
- Cached model loading and Streamlit task switching through Streamlit's testing
  API with a lightweight fake streaming runtime.
- Existing synchronous generation and task-building contracts as regression
  coverage.

After unit tests pass, two real smoke runs will use the local fine-tuned adapter
and `data/examples/story.txt`: one extraction and one Arabic-to-English
translation. Each run must produce at least one chunk before completion and a
final response that passes its corresponding schema. If CUDA is unavailable,
the same checks run on CPU with the expected longer duration explicitly
reported.

The final verification also runs all project tests and starts Streamlit in
headless mode long enough to confirm a healthy page response and absence of
startup exceptions.

## Dependencies and Commands

Streamlit will be isolated in a dedicated dependency group rather than added to
training, evaluation, or serving environments. Transformers and PEFT remain in
the model/evaluation dependency path.

The documented local command will be:

```powershell
uv run --group app streamlit run app.py
```

Environment documentation will cover the existing `HF_TOKEN`, the optional Hub
adapter source in `FINETUNED_ADAPTER_SOURCE`, and the optional `vllm` settings
in `VLLM_API_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL_ID`. Local defaults
continue to target the current adapter directory, `http://localhost:8000/v1`,
and `news-lora` so existing workflows retain their behavior.

## Deployment Implication

The UI/runtime separation supports two deployments without changing the page:

- A single Streamlit container that loads Qwen and the adapter directly.
- A CPU Streamlit frontend configured for the `vllm` provider and connected to
  a separately secured GPU inference service.

Hugging Face no longer offers Streamlit as a built-in SDK, so a future
Hugging Face Streamlit Space must use the Docker SDK. The local implementation
does not require Docker and remains the first verified target.

## Compatibility Guarantees

- No existing prompt strings or output schemas are copied into the UI.
- Existing synchronous `generate()` entry points remain available.
- Training configuration, datasets, evaluation reports, vLLM scripts,
  middleware, Chinese-token suppression, Locust workload, and output paths are
  unchanged.
- The user's existing uncommitted `hub_model_id` change is not included in this
  work.
- New frontend dependencies are opt-in through the application dependency
  group.
