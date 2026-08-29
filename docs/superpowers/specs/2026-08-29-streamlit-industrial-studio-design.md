# Streamlit Industrial Inference Studio Design

Date: 2026-08-29
Status: Approved direction; awaiting written-spec review
Target branch: `master`

## Outcome

Restyle the existing Streamlit application as a focused dark inference studio
and add a configuration-only sidebar. Preserve the current extraction and
translation prompts, genuine token streaming, response validation, model
caching, direct fine-tuned backend, and vLLM backend.

The reference code supplies the intended product shape, not implementation
logic. Its synchronous `/v1/completions` request, duplicated prompt handling,
fabricated connection badge, history tab, architecture tab, and session metrics
will not be copied.

## Product and visual direction

Problem: give a developer or evaluator a clear laptop-scale workspace for
running the fine-tuned Arabic news model and seeing output as it is generated.

Visual anchor: Industrial. The product is a technical inference workspace, so
the interface will use a pitch-black or warm-black surface, monospace UI type,
flat one-pixel borders, tabular values, and one semantic green signal color.
There will be no gradient, glow, shadow, rounded card stack, decorative emoji,
or fabricated telemetry.

Signature differentiator: the result column behaves as a live proof console.
It identifies the active backend and model, shows an honest waiting state until
the first chunk arrives, and then renders genuine model chunks through
`st.write_stream` with Streamlit's typing cursor.

Positioning decisions:

- Narrative role: working inference console, not a marketing page.
- Viewing distance: approximately one metre on a laptop or desktop.
- Visual temperature: calm, technical, and authoritative.
- Capacity: fixed-width configuration sidebar plus a two-column task workspace;
  the columns stack at narrow widths.

## Scope

### Included

- Configuration-only sidebar.
- Backend selection between vLLM and direct fine-tuned inference.
- vLLM endpoint, model ID, and masked API-key controls.
- vLLM temperature and maximum-output-token controls.
- A real, user-triggered vLLM connection check with `connected`, `unavailable`,
  and `not checked` states.
- Extraction and Translation tabs.
- Paste and UTF-8 TXT/Markdown upload inputs.
- Source and target language controls for Translation.
- Genuine progressive output for both tasks.
- Structured validated results plus an optional raw JSON view.
- Responsive styling and automatic Arabic/Latin text direction.
- Existing error states for input, connectivity, model loading, GPU memory,
  partial generation, and invalid schema output.

### Excluded

- History and Architecture tabs.
- Session request, latency, or token metrics.
- Docker configuration.
- A new FastAPI or Uvicorn gateway.
- Changes to training, evaluation, Locust, existing API behavior, prompt text,
  or vLLM server launch scripts.

## Interface structure

### Sidebar

The sidebar contains only settings that affect inference:

1. Product name and a compact backend-status line.
2. Backend selector: `vLLM` or `Direct fine-tuned model`.
3. For vLLM:
   - API base URL, defaulting to `VLLM_API_BASE_URL`.
   - Model ID, defaulting to `VLLM_MODEL_ID`.
   - API key in a password field, defaulting to `VLLM_API_KEY` without ever
     rendering the secret value outside the input.
   - Temperature and maximum-output-token controls.
   - `Check connection` action. It queries the configured server and verifies
     that the configured model ID is served; the UI never claims connectivity
     before this succeeds.
4. For direct inference:
   - Fine-tuned adapter source and base model shown as read-only information.
   - Generation stays on the project's tested deterministic defaults, avoiding
     model reloads when a visual control changes.

Environment values remain the reproducible defaults; sidebar values are
session-local overrides and are never written to disk.

### Main workspace

The header uses the real product name `ArabLLM Inference Studio` and one concise
description of the two supported tasks.

Two tabs provide separate, explicit workflows:

- Extraction: story paste/upload controls on the left and live model output on
  the right.
- Translation: language selectors and story paste/upload controls on the left,
  with live translated output on the right.

Each tab uses unique Streamlit widget keys. Submitting one task does not mutate
the other task's inputs or last result.

## Architecture and data flow

```text
Sidebar settings
      |
      v
Streamlit task tab -> resolve_story_input -> NewsStreamingController
                                           |-> existing extraction prompts
                                           |-> existing translation prompts
                                           v
                          direct Transformers/PEFT or vLLM streaming runtime
                                           |
                                           v
                          st.write_stream -> schema validation -> result view
```

The UI remains an adapter over existing application boundaries:

- `src/ui/streamlit_app.py` owns page composition and rendering.
- A small UI settings module owns the immutable, validated session settings and
  sidebar rendering data.
- `NewsStreamingController` continues to own task-to-prompt routing and schema
  validation.
- Model classes continue to own real generation streaming.
- The model factory accepts explicit vLLM connection overrides rather than
  mutating process-wide environment variables.

`st.cache_resource` keys include backend connection identity but not transient
presentation state. Changing a vLLM endpoint or model creates the appropriate
client resource. Ordinary Streamlit reruns reuse the existing resource. Direct
model weights remain cached and do not reload when users switch tasks.

## Streaming behavior

After submission:

1. Validate pasted/uploaded input before loading or contacting a model.
2. Show `Preparing inference…` while the cached runtime/client is resolved.
3. Show `Waiting for the first token…` until the first genuine chunk arrives.
4. Send the runtime iterator directly to `st.write_stream`.
5. Change status to `Streaming response…` after the first chunk.
6. Concatenate the same emitted chunks for validation; do not make a second
   generation request.
7. On completion, validate the full JSON and render the structured result.

No completed response will be replayed character by character. Any cursor or
typing motion is presentation around the real streamed chunks.

## Error handling and security

- Invalid or empty input stops before model/client construction.
- Connectivity failures identify the configured endpoint without exposing the
  API key.
- API keys remain in process/session memory and masked inputs; logs and stored
  history do not contain them.
- A partial streamed response remains available when generation fails.
- Invalid JSON or schema output is shown as a warning with raw output available
  for diagnosis.
- The UI will not expose a public vLLM port or add authentication infrastructure;
  deployment should keep vLLM on a private network.

## Compatibility

The change must not alter:

- extraction or translation prompt builders;
- model response schemas;
- synchronous model `generate` behavior;
- vLLM server middleware or launch commands;
- API, evaluation, training, Locust, or load-test code;
- the user's unrelated working-tree changes.

## Verification

- Unit tests for settings defaults, validation, masking behavior, and connection
  status mapping.
- Streamlit AppTest coverage for initial rendering, both task tabs, conditional
  sidebar controls, empty-input validation, and lazy model loading.
- Existing controller, Qwen streaming, vLLM streaming, configuration, and input
  tests remain green.
- Python compilation and Streamlit health endpoint checks.
- Browser inspection at desktop and mobile widths, including both tabs and the
  sidebar.
- A real vLLM end-to-end generation check when the local WSL server is available;
  otherwise retain the already-passing real direct-model smoke evidence and
  report the unavailable external server explicitly.

## Acceptance criteria

- The application opens with no model loaded and no false `connected` badge.
- The sidebar contains configuration only.
- Extraction and Translation both stream genuine runtime chunks visibly.
- Configuration overrides reach vLLM without modifying `src/.env`.
- Direct model weights are not reloaded on ordinary UI interactions.
- Output validates and renders through the existing Pydantic schemas.
- Desktop and mobile layouts remain legible with Arabic and Latin text.
- No Docker, History tab, Architecture tab, fabricated metrics, or separate
  Uvicorn service is introduced.
