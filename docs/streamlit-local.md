# Streamlit application

The Streamlit UI has two backends behind the same controller and task prompts:

```text
Streamlit UI -> NewsStreamingController -> extraction/translation prompts
                                      -> direct Transformers + PEFT model
                                      -> vLLM OpenAI-compatible API
```

Use the direct `finetuned` backend for local development or one-user demos. Use
the `vllm` backend on an NVIDIA GPU for lower latency and concurrent users. The
response is streamed from the selected backend; the UI does not fake the effect
by revealing an already-complete answer.

No separate Uvicorn process is required. Streamlit owns its web server, and
vLLM owns its OpenAI-compatible inference server. Add a FastAPI/Uvicorn gateway
only if a later deployment needs custom authentication, quotas, or a public API
that is independent of the UI.

## Install

The project requires Python 3.12 and `uv`.

```powershell
uv sync --cache-dir .uv-cache --group app --group serve
Copy-Item src/.env.example src/.env
```

Set `HF_TOKEN` in `src/.env`. The example configuration loads the fine-tuned
adapter from `marouaHattab/ArabLLM-news`. To use the checked-out adapter instead,
set `FINETUNED_ADAPTER_SOURCE=outputs/models/news-finetune`.

## Run with the model inside Streamlit

Keep these values in `src/.env`:

```dotenv
NEWS_MODEL_PROVIDER=finetuned
FINETUNED_ADAPTER_SOURCE=marouaHattab/ArabLLM-news
HF_TOKEN=your_hugging_face_token
```

Then start the application:

```powershell
uv run --cache-dir .uv-cache --group app streamlit run app.py
```

The model is cached with `st.cache_resource`, so Streamlit reruns do not reload
the weights. A CUDA GPU is selected automatically when available; otherwise the
model runs on CPU and generation will be much slower.

## Sidebar configuration

`src/.env` supplies reproducible defaults. The sidebar overrides them for the
current browser session only and never writes `src/.env`.

- **Backend** chooses `vLLM` or `Direct fine-tuned model`.
- For vLLM, set the API base URL, model ID, temperature, and maximum output
  tokens. Leave **API key override** blank to use `VLLM_API_KEY`. The field
  never displays the environment secret.
- **Check connection** queries the configured server and confirms the model ID
  is served. Status stays `Not checked` until that action succeeds. The other
  states are `Connected` and `Unavailable`.
- Direct inference shows the adapter source and base model as read-only
  information. Generation stays on the project's deterministic defaults.

Extraction and Translation are separate tabs. Submitting one task does not
change the other task's inputs or last result.

## Run with vLLM

The included vLLM launcher uses WSL2/Linux because vLLM is not installed on
native Windows. It expects the LoRA adapter at
`outputs/models/news-finetune` and an NVIDIA GPU visible inside WSL.

In terminal 1:

```powershell
.\deployment\vllm\serve.ps1
```

In `src/.env`, select the API backend:

```dotenv
NEWS_MODEL_PROVIDER=vllm
VLLM_API_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-vllm
VLLM_MODEL_ID=news-lora
```

In terminal 2, check readiness and start Streamlit:

```powershell
uv run --cache-dir .uv-cache --group serve python -m src.workflows.check_vllm --wait
uv run --cache-dir .uv-cache --group app streamlit run app.py
```

Stop the inference server with:

```powershell
.\deployment\vllm\stop.ps1
```

## Hugging Face Spaces without Docker

Hugging Face deprecated its built-in Streamlit Spaces SDK on April 30, 2025 and
now documents Streamlit Spaces through the Docker SDK. Therefore this
Docker-free Streamlit version should run locally or on a VM. If deployment must
be a native Hugging Face Space without Docker, use a Gradio frontend and keep
the same controller/backend boundary. ZeroGPU is also limited to Gradio Spaces.

- Streamlit Spaces: https://huggingface.co/docs/hub/main/spaces-sdks-streamlit
- ZeroGPU: https://huggingface.co/docs/hub/main/en/spaces-zerogpu

