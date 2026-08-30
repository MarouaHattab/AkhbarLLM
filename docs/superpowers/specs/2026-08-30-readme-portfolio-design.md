# AkhbarLLM README Portfolio Design

Date: 2026-08-30

## Objective

Rewrite the root README as a layered project case study that serves two audiences:

1. AI/ML recruiters and hiring managers must understand the project's value, scope, ownership, and engineering depth in under one minute.
2. Technical and open-source readers must be able to understand the architecture, reproduce the main workflows, and inspect the evidence without leaving the README.

The rewrite must make the project feel important through clear problem framing, concrete scope, and engineering judgment. It must not exaggerate the current evaluation evidence.

## Current-State Audit

The existing README contains strong technical material but presents it in a documentation-first order:

- The opening communicates the full stack, but the phrases `match teacher-quality` and `zero per-request cost` exceed what the published evidence supports.
- The architecture diagrams are useful, but project ownership and engineering contributions appear near the end.
- The base-versus-fine-tuned comparison is compelling but is based on one documented story, not an aggregate held-out evaluation.
- The full translation JSON consumes substantial vertical space and delays later sections.
- Training, serving, evaluation, and load-test commands are detailed, but they interrupt the portfolio narrative.
- The `Skills`, `Stack`, and `Numbers` sections repeat facts that should be connected directly to delivered components.
- The bundled LLaMA-Factory source can make authorship ambiguous unless the README distinguishes upstream code from project-specific work.
- The current validation split is sample-level rather than fully grouped by source story; this must be disclosed as a limitation rather than hidden behind validation loss.

## Editorial Principles

### Evidence before adjectives

Use numbers and shipped components to establish importance. Avoid unsupported superlatives such as `teacher-quality`, `production-grade`, or `zero cost`.

### Ownership before tool lists

Replace the generic skills inventory with a `What I built` section. Each claimed skill must map to a repository component or workflow.

### Progressive disclosure

Keep the first screen concise. Move long output samples and detailed command sequences into clearly labeled subsections or HTML `details` blocks.

### Two reading paths

The upper portion is a portfolio case study. The lower portion is a reproducibility guide. Readers must not need separate READMEs to understand the complete project.

### Transparent evaluation

Describe the current comparison as a qualitative smoke test. Separate format/schema correctness from semantic model quality, and state the current methodological limitations explicitly.

## README Information Architecture

The rewritten README will use this order.

### 1. Hero

Content:

- Project name: `AkhbarLLM`.
- One accurate impact statement:

  > An end-to-end Arabic news NLP system that distills structured labels from a teacher model, fine-tunes Qwen2.5-1.5B with LoRA, and serves schema-validated streaming inference locally with vLLM.

- One compact implementation sentence covering OpenAI `o4-mini`, LLaMA-Factory, Kaggle 2x T4, vLLM, Pydantic, and Streamlit.
- Correctly labeled links for the adapter, W&B training run, and video demo.
- The existing `docs/assets/project-overview.png` image.

The three currently modified diagram PNGs are user-owned changes and must not be regenerated or overwritten during the README rewrite.

### 2. Why this project matters

Explain the practical problem in one short paragraph:

- Arabic news arrives as unstructured text.
- Downstream systems need predictable titles, keywords, summaries, categories, entities, and translations.
- Paid teacher models are useful for labeling but create ongoing API dependency at inference time.
- The project demonstrates converting that capability into a small self-hosted adapter with schema validation.

Use `no third-party API charge at inference` rather than `zero per-request cost`.

### 3. Project at a glance

Show a compact metrics table near the top:

| Area | Evidence |
| --- | --- |
| Source data | 2,400 Arabic news stories |
| SFT data | 2,766 records; 2,700 train and 66 validation |
| Base model | Qwen2.5-1.5B-Instruct |
| Adaptation | LoRA rank 64 across all linear layers |
| Training | 3 epochs on Kaggle 2x NVIDIA T4 |
| Tasks | Structured extraction and translation |
| Serving | Local vLLM OpenAI-compatible API with streaming |
| Validation | Pydantic `NewsDetails` and `TranslatedStory` schemas |

Do not present the 20-user load-test configuration as a performance outcome unless measured RPS, latency, and failure-rate results are published.

### 4. What I built

Map ownership to concrete deliverables:

- Teacher-distillation and JSONL dataset workflows.
- Schema-guided extraction and translation prompts.
- LLaMA-Factory formatting, registration, and LoRA configuration.
- Model-provider abstraction for base Qwen, fine-tuned Qwen, Gemini, OpenAI, and vLLM.
- Evaluation and benchmark command-line workflows.
- vLLM deployment scripts, LoRA hot-loading, and CJK token-suppression middleware.
- Pydantic validation and streaming controllers.
- Streamlit interface and Locust load-testing workflow.

Add one sentence clarifying that `LlamaFactor/` is the upstream LLaMA-Factory checkout used by the project, while the project-specific orchestration lives in `src/`, `configs/`, `deployment/`, and `tests/`.

### 5. Documented result

Title the section `Documented smoke test: base Qwen vs AkhbarLLM`.

Requirements:

- State that both models receive the same story and task prompts.
- Retain the concise comparison table.
- Describe JSON parsing and Pydantic schema validation as deterministic checks.
- Describe language quality, entity grounding, and translation completeness as observations from this example, not aggregate scores.
- Put full extraction and translation examples inside separate `details` blocks.
- Link to the committed evaluation report files.

### 6. Architecture

Show and explain the three diagrams in this order:

1. Project overview.
2. Training pipeline.
3. Inference pipeline.

Use brief explanatory paragraphs. Do not repeat all labels already visible in the images.

### 7. Engineering decisions

Use a compact table connecting decisions to rationale:

| Decision | Rationale |
| --- | --- |
| Teacher distillation | Pay for labeling once and reuse structured supervision |
| LoRA | Adapt a small model without full-parameter fine-tuning |
| JSON Schema in prompts | Make the target structure explicit to every provider |
| Pydantic validation | Reject malformed or schema-incompatible output deterministically |
| vLLM OpenAI-compatible API | Support streaming and standard client integration |
| CJK token suppression | Prevent unwanted Chinese vocabulary from appearing in Arabic output |
| Provider factory | Compare local and hosted models through one interface |

### 8. Quick start

Provide the shortest local application path:

1. Copy `src/.env.example` to `src/.env`.
2. Install the application dependency group.
3. Select the model provider.
4. Start Streamlit.

State the GPU/model requirements for direct fine-tuned loading and vLLM. Do not imply that the local adapter files are committed when the directory contains only a placeholder in Git.

### 9. Reproduce the pipeline

Retain the detailed technical workflows under clear subsections:

1. Generate teacher-labeled data.
2. Format and register LLaMA-Factory datasets.
3. Fine-tune the LoRA adapter.
4. Evaluate model providers.
5. Serve with vLLM.
6. Run the Streamlit UI.
7. Run the Locust load test.

Keep commands executable and use `powershell`, `bash`, or `console` fences rather than `python` fences for shell commands.

### 10. Schemas and repository map

Retain concise schema examples and the project tree. Remove generated or ignored artifacts from the tree unless the text clearly states they are created locally.

### 11. Evaluation scope and limitations

State all of the following clearly:

- The committed base-versus-fine-tuned report is a one-story qualitative smoke test.
- JSON validity and schema validity do not measure semantic correctness.
- The current train/validation split is sample-level, so source-story overlap across tasks is possible.
- Aggregate category accuracy, entity precision/recall/F1, translation quality, latency percentiles, throughput, and failure rate are not yet published.
- Local inference removes hosted-model API charges but still consumes local compute and electricity.

Frame these as the next evaluation milestones, not as hidden defects.

### 12. Stack and acknowledgements

End with one consolidated stack table. Remove the separate generic `Skills` and duplicate `Numbers` sections after their evidence has been integrated earlier.

Credit:

- Qwen2.5-1.5B-Instruct as the base model.
- LLaMA-Factory as the upstream fine-tuning framework.
- OpenAI `o4-mini` as the labeling teacher.
- vLLM, PEFT, Pydantic, Streamlit, W&B, and Hugging Face for their specific roles.

Do not add a license claim while the repository has no root license file.

## Claim and Link Corrections

The implementation must:

- Remove or rewrite `match teacher-quality`.
- Replace `zero per-request cost` with `no third-party API charge at inference` or equivalent qualified wording.
- Label the Google Drive asset as a `Video demo`, not a live `Streamlit` deployment.
- Verify the public Hugging Face adapter URL and use one canonical repository identifier consistently in the README.
- Avoid describing a load-test configuration as a successful benchmark result.
- Avoid presenting validation loss as proof of end-task quality.

## Scope

### In scope

- Rewrite `README.md` according to this information architecture.
- Preserve and embed the three current diagram PNGs.
- Correct claims, labels, code-fence languages, local paths, and duplicate sections.
- Keep a complete technical reproduction path in the root README.

### Out of scope

- Retraining the model.
- Regenerating datasets.
- Redesigning or overwriting the diagram PNGs.
- Adding new benchmark results that have not been measured.
- Fixing the dataset split or implementing a new evaluation suite.
- Adding a software license without an explicit licensing decision.

## Verification Plan

Before declaring the rewrite complete:

1. Check every local README image and file link exists.
2. Verify external adapter, W&B, and video-demo links.
3. Confirm all numerical claims against repository data and training configuration.
4. Confirm shell commands match current CLI arguments and dependency groups.
5. Scan for unsupported phrases including `teacher-quality`, `zero per-request cost`, and unqualified `production` claims.
6. Confirm the three diagram files are not modified by the README implementation.
7. Check Markdown code fences are balanced and use appropriate languages.
8. Render or preview the README and inspect heading hierarchy, table readability, image scaling, and collapsible examples.
9. Run the existing non-destructive test suite as a repository health signal, while reporting skipped model smoke tests honestly.

## Acceptance Criteria

- A hiring reviewer can identify the problem, impact, ownership, architecture, and strongest engineering skills within the first 100 lines.
- A technical reader can find setup, training, evaluation, serving, and load-testing commands from the same README.
- Every major skill claim is tied to a concrete repository component.
- The documented result is useful but accurately scoped as a smoke test.
- Limitations are explicit and do not contradict claims elsewhere.
- Long JSON examples no longer dominate the main reading flow.
- Repeated `Skills`, `Stack`, and `Numbers` content is consolidated.
- The existing modified diagram PNGs remain untouched.
