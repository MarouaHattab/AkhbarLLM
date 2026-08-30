# AkhbarLLM

![AkhbarLLM overview](docs/assets/project-overview.png)

**AkhbarLLM** is a local Arabic news model. It takes an Arabic story and returns **raw, schema-valid JSON** for:

1. **Extraction** — Arabic title, keywords, summary, category, named entities (`NewsDetails`)
2. **Translation** — title + full body (`TranslatedStory`)

It is a **LoRA adapter (rank 64)** on [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). It was **trained on Kaggle with 2× NVIDIA T4 (T4 × 2)**. Training is tracked in [Weights & Biases](https://wandb.ai/marouahattab3-cole-polytechnique/llamafactory?nw=nwusermarouahattab3). The adapter is on Hugging Face: [marouaHattab/ArabLLM-news](https://huggingface.co/marouaHattab/ArabLLM-news).

Serve with **vLLM in WSL**. Use **Streamlit on Windows**. Load-test with **Locust**.

| | |
| --- | --- |
| Adapter | [huggingface.co/marouaHattab/ArabLLM-news](https://huggingface.co/marouaHattab/ArabLLM-news) |
| Training logs | [W&B · llamafactory · T4 × 2](https://wandb.ai/marouahattab3-cole-polytechnique/llamafactory?nw=nwusermarouahattab3) |
| Hardware | Kaggle **GPU T4 × 2** |

---

## Why fine-tune

Base Qwen 1.5B is a general chat model. On Arabic news it is **bad at Arabic** and **bad at JSON**:

- Titles and keywords come out in **English**, not Arabic.
- Output is wrapped in `` ```json `` fences, so parsing fails.
- Entity lists dump schema enum names (`Person`, `Location`, `Disease`) as if they appeared in the story.
- Translation is a short English sentence, not the full article.
- Chinese tokens from Qwen’s vocab can leak into the output.

AkhbarLLM distills teacher JSON from OpenAI `o4-mini`, then LoRA-tunes Qwen on **Kaggle T4 × 2** so the student emits Arabic JSON that validates.

Same story for both evals: `data/examples/story.txt` (فوربس / شاين إنيت / العلاقة بالمال). Reports: `outputs/Evaluation/qwen-base-evaluation.txt` and `outputs/Evaluation/qwen-finetuned-evaluation.txt`.

| Check | Base Qwen 1.5B | AkhbarLLM (fine-tuned) |
| --- | --- | --- |
| Extraction language | English | **Arabic** |
| Extraction JSON | Fail (markdown fence) | **Pass** |
| Extraction schema | Fail | **Pass** |
| Translation JSON | Fail (markdown fence) | **Pass** |
| Translation schema | Fail | **Pass** |
| Translation length | One sentence | Full article |

### Base Qwen — extraction (broken)

English title, English keywords, invented entities, markdown fence. **JSON valid: False. Schema valid: False.**

```json
{
  "story_title": "Family Influence on Financial Relationships",
  "story_keywords": ["Forbes", "Financial Therapy Association", "Money Genogram"],
  "story_category": "economy",
  "story_entities": [
    { "entity_value": "Forbes", "entity_type": "organization" },
    { "entity_value": "A", "entity_type": "not_specified" },
    { "entity_value": "Person", "entity_type": "person-male" },
    { "entity_value": "Location", "entity_type": "location" },
    { "entity_value": "Disease", "entity_type": "disease" },
    { "entity_value": "Not Specified", "entity_type": "not_specified" }
  ]
}
```

### AkhbarLLM — extraction (valid)

Arabic title and keywords, entities from the story. **JSON valid: True. Schema valid: True.**

```json
{
  "story_title": "دور العائلة في تشكيل علاقة الأفراد بالمال",
  "story_keywords": ["العائلة", "العلاقة بالمال", "الشخصية المالية", "الثروة", "الإنفاق"],
  "story_category": "economy",
  "story_entities": [
    { "entity_value": "فوربس", "entity_type": "organization" },
    { "entity_value": "شاين إنيت", "entity_type": "person-male" },
    { "entity_value": "رابطة العلاج المالي", "entity_type": "organization" },
    { "entity_value": "Money Genogram", "entity_type": "artifact" }
  ]
}
```

### Base Qwen — translation (broken)

One English sentence inside a markdown fence. **JSON valid: False.**

```json
{
  "translated_title": "Forbes Magazine Reveals Family Plays a Central Role in Forming Individuals' Financial Relationships",
  "translated_content": "According to Forbes magazine, family plays a central role in shaping individuals' financial relationships, as these relationships are influenced by inherited behavioral patterns across generations."
}
```

### AkhbarLLM — translation (valid)

Full article, no fence. **JSON valid: True. Schema valid: True.**

```json
{
  "translated_title": "The Role of Family in Financial Relationships",
  "translated_content": "Forbes magazine reported that family plays a central role in shaping individuals' relationship with money… The Financial Therapy Association developed a tool called the Money Genome Map (Genogram)…"
}
```

Trainer val loss on Kaggle T4 × 2 (effective batch 8): **0.3619** at step 1000. Best val loss **0.346** at step 600.

---

## Architecture

Redraw: `python docs/assets/render_architecture.py`

### Training (Kaggle T4 × 2)

![Training pipeline](docs/assets/training-pipeline.png)

Raw Arabic news → teacher `o4-mini` labels → SFT JSONL → LLaMA-Factory `train.json` / `val.json` → **LoRA SFT on Kaggle T4 × 2** → rank-64 adapter. W&B logs the run. Hugging Face hosts [marouaHattab/ArabLLM-news](https://huggingface.co/marouaHattab/ArabLLM-news). The 1.5B base stays frozen.

### Inference

![Inference pipeline](docs/assets/inference-pipeline.png)

Streamlit (Windows) or Locust talks to **vLLM in WSL** (`Qwen2.5-1.5B` + LoRA `news-lora` + CJK suppressor) at `http://localhost:8000/v1`. Extraction and translation JSON are checked with Pydantic.

---

## Setup

Python **3.12**. Use [uv](https://docs.astral.sh/uv/).

```powershell
copy src\.env.example src\.env
```

```env
HF_TOKEN=hf_...
WANDB_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
NEWS_MODEL_PROVIDER=vllm
FINETUNED_ADAPTER_SOURCE=marouaHattab/ArabLLM-news
VLLM_API_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-vllm
VLLM_MODEL_ID=news-lora
```

```powershell
uv sync --group app --group evaluation --group serve --group train --group distillation
```

---

## All scripts

Run every workflow with `python -m …` from the repo root.

| Script | Command |
| --- | --- |
| Distill labels (`o4-mini` → `data/datasets/sft.jsonl`) | `uv run --group distillation python -m src.workflows.generate_distillation_dataset` |
| Format SFT → LLaMA-Factory JSON | `uv run --group train python -m src.workflows.format_finetuning_dataset` |
| Register datasets in LLaMA-Factory | `uv run --group train python -m src.workflows.register_llamafactory_datasets --llamafactory-dir LlamaFactor` |
| Prepare train (HF + W&B login, copy YAML) | `uv run --group train python -m src.workflows.prepare_finetuning --llamafactory-dir LlamaFactor` |
| Train on Kaggle **T4 × 2** | `cd LlamaFactor` then `llamafactory-cli train examples/train_lora/news_finetune.yaml` |
| Eval base Qwen | `uv run --group evaluation python -m src.workflows.evaluate_models --model qwen --task both --output outputs/Evaluation/qwen-base-evaluation.txt` |
| Eval AkhbarLLM | `uv run --group evaluation python -m src.workflows.evaluate_models --model finetuned --task both --output outputs/Evaluation/qwen-finetuned-evaluation.txt` |
| Eval Gemini / OpenAI / all | `--model gemini` / `--model openai` / `--model all` |
| Latency benchmark | `uv run --group evaluation python -m src.workflows.benchmark_models --model both --samples 30` |
| Start vLLM (Windows → WSL) | `.\deployment\vllm\serve.ps1` |
| Start vLLM (inside WSL) | `bash deployment/vllm/serve.sh` |
| Stop vLLM | `.\deployment\vllm\stop.ps1` or `bash deployment/vllm/stop.sh` |
| Wait until `news-lora` is up | `uv run --group serve python -m src.workflows.check_vllm --wait` |
| Eval through vLLM | `uv run --group serve python -m src.workflows.infer_vllm --task both` |
| Streamlit UI | `uv run --group app streamlit run app.py` |
| Locust load test | `uv run --group serve locust -f tests/load/locustfile.py --headless --host=http://localhost:8000 -u 20 -r 1 -t 60s --html=outputs/LoadTesting/locust-results.html` |
| Locust token summary | `uv run --group serve python -m src.workflows.analyze_vllm_load` |

Config file for training: `configs/llamafactory/news_finetune.yaml`.

---

## Format data

```powershell
uv run --group train python -m src.workflows.format_finetuning_dataset
```

Reads `data/datasets/sft.jsonl`. Each line:

```json
{
  "story": "…Arabic article…",
  "task": "Extract the story details into a JSON object according to the provided schema.",
  "output_schema": { },
  "response": { "story_title": "…", "story_keywords": ["…"] }
}
```

Writes:

| File | Size |
| --- | --- |
| `data/datasets/llama_factory/train.json` | 2700 rows |
| `data/datasets/llama_factory/val.json` | 66 rows |
| `data/datasets/llama_factory/dataset_info.json` | column map |

`instruction` = `# Story` + `# Task` + `# Output Schema` + `# Output JSON:`. `output` = teacher JSON. Rebuild labels (paid API):

```powershell
uv run --group distillation python -m src.workflows.generate_distillation_dataset
```

---

## Train on Kaggle (T4 × 2)

Training **must** use two T4 GPUs (Kaggle accelerator **GPU T4 × 2**). That is how the published adapter was trained.

| Setting | Value |
| --- | --- |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA SFT, rank **64**, all linear layers |
| Hardware | **Kaggle T4 × 2**, DDP |
| Batch | 1 per GPU × 4 accum × 2 GPUs = **8** |
| Epochs / LR | 3 / `1e-4` cosine, warmup 0.1 |
| Precision | FP16 + gradient checkpointing |
| Context | `cutoff_len: 3500` |
| W&B | `report_to: wandb`, run `newsx-qwen2.5-1.5b-lora` |
| Hub | `push_to_hub: true` → [marouaHattab/ArabLLM-news](https://huggingface.co/marouaHattab/ArabLLM-news) |

```powershell
uv run --group train python -m src.workflows.prepare_finetuning --llamafactory-dir LlamaFactor
```

On Kaggle (GPU T4 × 2, internet on):

```bash
cd LlamaFactor
llamafactory-cli train examples/train_lora/news_finetune.yaml
```

Watch [W&B](https://wandb.ai/marouahattab3-cole-polytechnique/llamafactory?nw=nwusermarouahattab3). Adapter files: `outputs/models/news-finetune/` (`adapter_config.json`, `adapter_model.safetensors`).

---

## Evaluate

```powershell
uv run --group evaluation python -m src.workflows.evaluate_models --model qwen --task both --output outputs/Evaluation/qwen-base-evaluation.txt
uv run --group evaluation python -m src.workflows.evaluate_models --model finetuned --task both --output outputs/Evaluation/qwen-finetuned-evaluation.txt
```

`--task` is `extraction`, `translation`, or `both`. `--story` is any UTF-8 file. `--model` is `qwen`, `finetuned`, `gemini`, `openai`, or `all`.

---

## Serve vLLM on WSL (Windows)

vLLM 0.7.2 is Linux-only. WSL2 + Ubuntu + NVIDIA Windows driver (`nvidia-smi` inside Ubuntu). `uv` inside WSL. Adapter in `outputs/models/news-finetune/` or download [marouaHattab/ArabLLM-news](https://huggingface.co/marouaHattab/ArabLLM-news).

```powershell
.\deployment\vllm\serve.ps1
```

```bash
bash deployment/vllm/serve.sh
```

Binds `0.0.0.0:8000`, model id **`news-lora`**, CJK suppressor on. Linux venv: `~/.cache/news-finetuning/venv`.

```powershell
uv run --group serve python -m src.workflows.check_vllm --wait
.\deployment\vllm\stop.ps1
```

---

## Streamlit

```powershell
# src/.env → NEWS_MODEL_PROVIDER=vllm
uv run --group app streamlit run app.py
```

Direct GPU load (no vLLM): `NEWS_MODEL_PROVIDER=finetuned` and `FINETUNED_ADAPTER_SOURCE=outputs/models/news-finetune`.

---

## Locust

vLLM must already be running.

```powershell
New-Item -ItemType Directory -Force -Path outputs\LoadTesting | Out-Null
uv run --group serve locust -f tests/load/locustfile.py --headless --host=http://localhost:8000 -u 20 -r 1 -t 60s --html=outputs/LoadTesting/locust-results.html
uv run --group serve python -m src.workflows.analyze_vllm_load
```

Open `outputs/LoadTesting/locust-results.html` in Chrome or Edge. Interactive UI: same locust command without `--headless`, then `http://localhost:8089`.

---

## Layout

```
app.py                                 Streamlit
configs/llamafactory/news_finetune.yaml
data/raw/news-sample.jsonl             2400 unlabeled stories
data/datasets/sft.jsonl                teacher SFT (2766)
data/datasets/llama_factory/           train.json / val.json
data/examples/story.txt                eval story
deployment/vllm/                       serve.ps1, serve.sh, stop.*
docs/assets/                           overview + pipeline diagrams
outputs/Evaluation/                    base vs fine-tuned reports
outputs/models/news-finetune/          local LoRA
src/workflows/                         all python -m scripts
tests/load/locustfile.py
LlamaFactor/                           LLaMA-Factory used on Kaggle
```
