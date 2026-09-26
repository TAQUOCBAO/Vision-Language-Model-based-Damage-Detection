# Structural Damage Image-Text VLM Diagnosis

Offline pipeline to fine-tune **Qwen3-VL-8B-Instruct** (LoRA via LLaMA-Factory) so each image yields JSON:

```json
{"image_id": "...", "damage_categories": ["cracks", "..."], "description": "..."}
```

For a full English write-up of ideas, methods, training, evaluation, and results, see **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.  
For the official hidden-test protocol and results, see **[OFFICIAL_TEST_REPORT.md](OFFICIAL_TEST_REPORT.md)**.

---

## Choose your path

| Path | When to use | What you need |
|------|-------------|----------------|
| **[Part A — Inference only](#part-a--inference-only-no-retraining)** | Run predictions / light eval with the published LoRA. **No training.** | Dataset + Hugging Face LoRA + GPU |
| **[Part B — Train from scratch](#part-b--train-from-scratch-full-pipeline)** | Rebuild data tiers, bake-off, full retrain, then infer. | Dataset + ~30+ GiB free GPU VRAM + time |
| **[Part C — Official test set](#part-c--official-test-set-test-requirementsdocx)** | Score the trained model on the organizer hidden test (`001.jpg`–`110.jpg`) with the official Q1/Q2 prompt. | Official test folder + trained LoRA + GPU |

Most users who only want results should follow **Part A**.

---

## Common arguments (for beginners)

| Argument | Meaning |
|----------|---------|
| `--tier a\|b\|c\|full` | Which data recipe / train config (`a`/`b`/`c` = bake-off; `full` = retrain winner on all images). |
| `--adapter PATH` | Folder with the fine-tuned **LoRA** (`adapter_config.json` + `adapter_model.safetensors`). |
| `--model-dir PATH` | Folder with a **merged** full model (optional; from export `--merge`). |
| `--image PATH` / `--image-path PATH` | One image file. Repeat the flag for several files. |
| `--image-dir PATH` | Folder of images (used when you do not pass `--image`). |
| `--out PATH` | Where to write the main output JSON (predictions or metrics). |
| `--predictions-json PATH` | Existing predictions file to score (skip running the model again). |
| `--silver-json PATH` | Ground-truth labels used for F1 / METEOR. |
| `--load-in-4bit` | Load the 8B base in 4-bit (less VRAM; use only if GPU is tight). |
| `--load-in-8bit` | Same idea as 4-bit, slightly higher VRAM / quality than 4-bit. |
| `--limit N` | Only process the first `N` images (smoke tests). |
| `--repo-id USER/NAME` | Hugging Face model repo id for upload. |
| `--dataset-root PATH` | Dataset root containing `description.json` and `image/`. |
| `--gold-json PATH` | Organizer hidden-test gold labels (optional; official accuracy + METEOR). |

**Model path tip:** Prefer full-precision LoRA (best quality). Add `--load-in-4bit` only when VRAM is limited. Use the Hub adapter (`./models/damage-vlm-qwen3vl-8b-lora`) in Part A, or `outputs/runs/full_winner_v2` after Part B training.

---

## Download the dataset (Google Drive)

Updated competition corpus (**Dataset-Project 3-Updated**):

**https://drive.google.com/file/d/1n3JfQzqY77qnk9nKg4k3GkEMjajQXmt8/view**

### Option 1 — Browser

1. Open the link above and download `Dataset-Project 3-Updated.rar`.
2. Extract it so you get a folder that contains `description.json` and `image/`.
3. Place that content at:

```text
data_v2/dataset/description.json
data_v2/dataset/image/   # many .jpg files
```

Example (after you have the `.rar` in the repo root or `data_v2/`):

```bash
mkdir -p data_v2
# If the archive extracts to a folder named "dataset":
unrar x "data_v2/Dataset-Project 3-Updated.rar" data_v2/
# or: 7z x "data_v2/Dataset-Project 3-Updated.rar" -odata_v2/

# Confirm layout
ls data_v2/dataset/description.json data_v2/dataset/image | head
```

If the archive unpacks as `data_v2/dataset/dataset/...`, move the inner `description.json` and `image/` up one level so paths match the layout above.

### Option 2 — Command line (`gdown`)

```bash
pip install -U gdown
mkdir -p data_v2
gdown 'https://drive.google.com/uc?id=1n3JfQzqY77qnk9nKg4k3GkEMjajQXmt8' \
  -O "data_v2/Dataset-Project 3-Updated.rar"
unrar x "data_v2/Dataset-Project 3-Updated.rar" data_v2/
```

You need `unrar` or `7z` installed on the machine.

### Expected layout

```text
data_v2/dataset/
  description.json
  image/
    00002.jpg
    ...
```

Part A uses this folder for inference. Part B uses it for the full-winner retrain (`--dataset-root data_v2/dataset`). The older bake-off corpus (if present) lives under `data/dataset/` and is only needed for Part B steps 1–5.

---

# Part A — Inference only (no retraining)

Use the **published LoRA** from Hugging Face. You do **not** run LLaMA-Factory training.

## A0) Setup (inference)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"

# CUDA 12.8 wheels (this machine / driver 570). Do not use default cu130 wheels.
pip install --force-reinstall "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu128

pip install "qwen-vl-utils" "bitsandbytes" "huggingface_hub"

python -c "import torch; assert torch.cuda.is_available()"
nvidia-smi   # full precision ~16–22 GiB; or use --load-in-4bit (~4–8 GiB)
```

Then download the [dataset](#download-the-dataset-google-drive) into `data_v2/dataset/`.

## A1) Download the fine-tuned LoRA from Hugging Face

Published adapter (full-precision LoRA only, ~0.7 GiB):

**https://huggingface.co/nhantran214/damage-vlm-qwen3vl-8b-lora**

At runtime you still need the base model `Qwen/Qwen3-VL-8B-Instruct` (downloaded automatically by Transformers on first run). The **same** adapter folder works for:

- **Full precision (recommended):** no `--load-in-4bit`
- **Low VRAM:** add `--load-in-4bit` (requires `bitsandbytes`)

```bash
pip install -U "huggingface_hub"

huggingface-cli download nhantran214/damage-vlm-qwen3vl-8b-lora \
  --local-dir ./models/damage-vlm-qwen3vl-8b-lora
```

Optional login (private repos / higher rate limits):

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_xxx
```

You should see at least:

- `adapter_config.json`
- `adapter_model.safetensors`

## A2) Run inference

```bash
# Full precision (recommended, ~16–22 GiB VRAM)
python scripts/08_infer_full_winner.py \
  --adapter ./models/damage-vlm-qwen3vl-8b-lora \
  --image-dir data_v2/dataset/image

# One image
python scripts/08_infer_full_winner.py \
  --adapter ./models/damage-vlm-qwen3vl-8b-lora \
  --image data_v2/dataset/image/00002.jpg

# Low VRAM (~4–8 GiB)
python scripts/08_infer_full_winner.py --load-in-4bit \
  --adapter ./models/damage-vlm-qwen3vl-8b-lora \
  --image data_v2/dataset/image/00002.jpg
```

**Output:** `outputs/submissions/submission.json` (unless you set `--out`).

## A3) (Optional) Evaluate a few images

Scoring against silver labels needs labels built once from the dataset (no model training):

```bash
python scripts/07_build_full_winner.py --dataset-root data_v2/dataset

python scripts/11_eval_metrics.py \
  --adapter ./models/damage-vlm-qwen3vl-8b-lora \
  --image-dir data_v2/dataset/image \
  --limit 20
```

Or score an existing predictions file:

```bash
python scripts/11_eval_metrics.py \
  --predictions-json outputs/submissions/submission.json \
  --image-dir data_v2/dataset/image
```

---

# Part B — Train from scratch (full pipeline)

Run these steps **in order**. Optional steps are marked. Requires a strong GPU (target: NVIDIA RTX 5880 Ada ~49 GB; need ~30+ GiB free before training).

## B0) Setup (train)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"

# CUDA 12.8 wheels (this machine / driver 570). Do not use default cu130 wheels.
pip install --force-reinstall "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu128

# Python 3.14: override LLaMA-Factory's datasets pin
pip install "datasets==4.4.2" "qwen-vl-utils" "bitsandbytes" "huggingface_hub"

which llamafactory-cli
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
nvidia-smi   # need ~30+ GiB free before training
```

Train recipe: BF16 LoRA, freeze vision tower, batch 1 × accum 16, `image_max_pixels=524288`.

Download the [updated dataset](#download-the-dataset-google-drive) into `data_v2/dataset/`. For the bake-off (steps B1–B5) you also need the v1 corpus under `data/dataset/` (`description.json` + `image/`). If you only want to retrain the winner recipe on the updated set, you can skip to [B6](#b6-full-retrain-of-the-winner-on-data_v2) after placing a `outputs/evals/winner.json` (or reuse the Tier A recipe already wired for `--tier full`).

## B1) Build silver labels + Tier A data

Creates labels, frozen train/val split, and Tier A SFT jsonl from `data/dataset`.

```bash
python scripts/01_build_silver_and_sft.py
```

**Outputs:** `data/processed/silver_labels.json`, `data/processed/splits/v1.json`, `data/processed/tiers/a/{train,val}.jsonl`

## B2) Build Tier B and Tier C data

```bash
python scripts/02_build_tiers.py
```

**Outputs:** `data/processed/tiers/b/train.jsonl`, `data/processed/tiers/c/train.jsonl`

## B3) Train one tier (bake-off)

```bash
# Optional dry-run (prints command only)
python scripts/03_train_tier.py --tier a --dry-run

# Real train — repeat with --tier b and --tier c
python scripts/03_train_tier.py --tier a
```

| Arg | Meaning |
|-----|---------|
| `--tier a` | Train config `configs/llamafactory/tier_a_lora_sft.yaml` |
| `--dry-run` | Do not start training |
| `--log-file PATH` | Optional custom log path |

**Outputs:** `outputs/runs/tier_a/` (LoRA adapter), log under `outputs/logs/`

## B4) Infer on the holdout (val) set

Needs the adapter from B3.

```bash
python scripts/infer_holdout.py --tier a --adapter outputs/runs/tier_a
# Smoke: add --limit 2
```

| Arg | Meaning |
|-----|---------|
| `--tier a` | Tag used in output filename |
| `--adapter PATH` | LoRA folder from B3 |
| `--limit N` | Only first N val images |

**Output:** `outputs/preds/tier_a_val.json`

## B5) Score holdout + pick winner

```bash
python scripts/04_eval_holdout.py --tier a --predictions-json outputs/preds/tier_a_val.json
# Repeat for b and c after their infer steps

python scripts/05_select_winner.py
```

| Arg | Meaning |
|-----|---------|
| `--predictions-json PATH` | Predictions from B4 |
| `--tier a` | Name written into the eval report |

**Outputs:** `outputs/evals/a.json` (and `b.json` / `c.json`), then `outputs/evals/winner.json`

## B6) Full retrain of the winner on `data_v2`

Uses all images + the winning recipe (from `winner.json`).

```bash
python scripts/07_build_full_winner.py --dataset-root data_v2/dataset
python scripts/03_train_tier.py --tier full
```

| Arg | Meaning |
|-----|---------|
| `--dataset-root PATH` | Updated corpus (`description.json` + `image/`) |
| `--tier full` | Train `configs/llamafactory/full_winner_lora_sft.yaml` |

**Outputs:** `data/processed/full_winner/train.jsonl`, adapter `outputs/runs/full_winner_v2/`

## B7) Low-VRAM export (after B6)

Packs the LoRA into a small deploy folder + `.tar.gz`. Gzip does **not** lower VRAM by itself; use this package later with `--load-in-4bit`.

```bash
python scripts/09_export_low_vram.py --adapter outputs/runs/full_winner_v2
```

| Arg | Meaning |
|-----|---------|
| `--adapter PATH` | Full-precision LoRA from B6 |
| `--merge` | Optional: also write a merged full model folder (~16 GiB on disk) |

**Outputs:** `outputs/exports/full_winner_v2_lora/`, `outputs/exports/full_winner_v2_lora.tar.gz`

## B8) Inference (submission predictions)

```bash
# Recommended — full-precision LoRA (~16–22 GiB VRAM)
python scripts/08_infer_full_winner.py \
  --adapter outputs/runs/full_winner_v2 \
  --image-dir data_v2/dataset/image

# One or more images
python scripts/08_infer_full_winner.py \
  --adapter outputs/runs/full_winner_v2 \
  --image data_v2/dataset/image/00002.jpg

# VRAM-limited only — 4-bit + export from B7 (~4–8 GiB)
python scripts/08_infer_full_winner.py --load-in-4bit \
  --adapter outputs/exports/full_winner_v2_lora \
  --image data_v2/dataset/image/00002.jpg
```

| Arg | Meaning |
|-----|---------|
| `--adapter PATH` | LoRA folder to load |
| `--model-dir PATH` | Merged model folder (optional; skips LoRA) |
| `--image PATH` | Single image (repeatable) |
| `--image-dir PATH` | Image folder (ignored if `--image` is set) |
| `--out PATH` | Predictions JSON (default: `outputs/submissions/submission.json`) |
| `--load-in-4bit` | Low-VRAM backbone load |
| `--limit N` | Smoke test on first N images |

**Output:** `outputs/submissions/submission.json`

## B9) (Optional) Push model to Hugging Face Hub

```bash
huggingface-cli login    # or: export HF_TOKEN=hf_xxx

python scripts/10_push_to_hub.py \
  --repo-id YOUR_USER/damage-vlm-qwen3vl-8b-lora \
  --model-dir outputs/runs/full_winner_v2
```

| Arg | Meaning |
|-----|---------|
| `--repo-id USER/NAME` | Destination Hub repo |
| `--model-dir PATH` | Local folder to upload (LoRA or export package) |
| `--private` | Create/update as a private repo |
| `--dry-run` | List files only; do not upload |
| `--include-archive` | Also upload `<model-dir>.tar.gz` if present |

## B10) Evaluate F1 / METEOR / weighted score

```bash
# Recommended — full-precision adapter + image folder
python scripts/11_eval_metrics.py \
  --adapter outputs/runs/full_winner_v2 \
  --image-dir data_v2/dataset/image \
  --limit 20

# Specific images
python scripts/11_eval_metrics.py \
  --adapter outputs/runs/full_winner_v2 \
  --image data_v2/dataset/image/00002.jpg \
  --image data_v2/dataset/image/00006.jpg

# VRAM-limited only
python scripts/11_eval_metrics.py --load-in-4bit \
  --adapter outputs/exports/full_winner_v2_lora \
  --image-dir data_v2/dataset/image \
  --limit 20

# Score existing predictions only (no model load)
python scripts/11_eval_metrics.py \
  --predictions-json outputs/submissions/submission.json \
  --image-dir data_v2/dataset/image
```

| Arg | Meaning |
|-----|---------|
| `--adapter PATH` | LoRA folder used to generate predictions |
| `--model-dir PATH` | Merged model folder (optional) |
| `--image` / `--image-dir` | Which images to evaluate |
| `--silver-json PATH` | Gold labels (default: `data/processed/full_winner/silver_labels.json`) |
| `--predictions-json PATH` | Skip inference; only compute metrics |
| `--out PATH` | Metrics report JSON |
| `--preds-out PATH` | Predictions used for this eval |
| `--load-in-4bit` | Low-VRAM inference |
| `--limit N` | Evaluate only first N images |
| `--tag NAME` | Label stored inside the report |

**Outputs:** `outputs/evals/eval_<timestamp>.json`, `outputs/preds/eval_<timestamp>.json`  
Bake-off scores stay in `outputs/evals/{a,b,c,winner}.json`.

---

# Part C — Official test set (`Test Requirements.docx`)

This path **fully follows** the organizer specification in  
`data_v2/Dataset-Project 3-Test/dataset/Test Requirements.docx`.

**Specification (verbatim):**

- 110 images named `001.jpg` … `110.jpg`
- Required prompt template:
  - `Q1: Determine whether there is structural damage in the image?`
  - `Q2: Describe the damage characteristics based on the image?`
- Evaluation:
  1. **Accuracy** of predicted damage categories vs ground-truth labels
  2. **METEOR** of predicted descriptions vs human reference descriptions

Place (or keep) the official package at:

```text
data_v2/Dataset-Project 3-Test/dataset/
  Test Requirements.docx
  image/
    001.jpg
    ...
    110.jpg
```

## C1) Run official inference (Q1/Q2 prompt)

Uses the trained winner LoRA. Does **not** retrain.

Requires a working GPU (`torch.cuda.is_available()`). The script refuses to start on CPU unless you pass `--allow-cpu`.

```bash
# Recommended — full-precision LoRA on GPU
python scripts/12_test_official.py \
  --adapter outputs/runs/full_winner_v2

# Smoke test
python scripts/12_test_official.py --limit 2

# Low VRAM
python scripts/12_test_official.py --load-in-4bit \
  --adapter outputs/runs/full_winner_v2

# Resume a partial run
python scripts/12_test_official.py --resume
```

**Outputs:**

| File | Role |
|------|------|
| `outputs/submissions/official_test.json` | Official predictions (`image_id`, Q1 flag, categories, description) |
| `outputs/evals/official_test.json` | Machine-readable inventory + optional scores |
| `OFFICIAL_TEST_REPORT.md` | English report regenerated after each run |

Each prediction answers both official questions:

- **Q1** → `q1_has_structural_damage` + closed-set `damage_categories`
- **Q2** → technical `description` (orientation / severity / size)

## C2) Score against organizer gold (when available)

The public test zip does **not** include hidden ground-truth labels. When the organizer gold file is available:

```bash
python scripts/12_test_official.py \
  --predictions-json outputs/submissions/official_test.json \
  --gold-json path/to/official_gold.json
```

Gold JSON may be an object keyed by `image_id` or a list of records. Each record needs:

```json
{
  "image_id": "001",
  "damage_categories": ["cracks"],
  "description": "Human reference description."
}
```

Reported official metrics:

| Axis | Metric |
|------|--------|
| Q1 | Binary damage-presence accuracy |
| Q1 categories | Subset accuracy (exact set match) and label accuracy (1 − Hamming loss) |
| Q1 categories | Sample-average F1 (supplementary) |
| Q2 | METEOR vs reference descriptions |

---

## Project layout

| Path | Role |
|------|------|
| `configs/` | Labels, prompts, eval settings, LLaMA-Factory YAMLs |
| `src/damage_vlm/` | Data, metrics, postprocess, infer helpers |
| `scripts/` | Numbered pipeline entrypoints |
| `data/dataset/` | Corpus for bake-off (v1) |
| `data_v2/dataset/` | Updated corpus (from Google Drive) for infer / full retrain |
| `data_v2/Dataset-Project 3-Test/dataset/` | Official hidden test (`001.jpg`–`110.jpg` + `Test Requirements.docx`) |
| `data/processed/` | Generated SFT / splits / `full_winner/` |
| `models/` | Downloaded Hub LoRA (local; gitignored) |
| `outputs/` | Runs, preds, evals, submissions, logs, exports |

---

## Notes

- Offline by default (no cloud APIs in the main scripts).
- Closed label set: `configs/labels_v1.yaml`.
- Split seed: `data/processed/splits/v1.json`.
- Code and comments are English.
- Dataset images and `docs/` are gitignored; download the dataset from Drive and the LoRA from Hugging Face.
