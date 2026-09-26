# Official Test Report — Structural Damage Image–Text VLM

**Generated:** 2026-09-19 10:38 UTC  
**Specification:** `Test Requirements.docx` (110 images, official Q1/Q2 prompt, category accuracy + METEOR)

## Compliance checklist

| Requirement | Status |
|-------------|--------|
| 110 images named `001.jpg`–`110.jpg` | Yes (`/home/ai/ethan/Project3/data_v2/Dataset-Project 3-Test/dataset/image`) |
| Specified prompt Q1 used verbatim | `Q1: Determine whether there is structural damage in the image?` |
| Specified prompt Q2 used verbatim | `Q2: Describe the damage characteristics based on the image?` |
| Task 1: damage category classification | `damage_categories` from closed 10-class set |
| Task 2: fine-grained characteristic description | `description` (orientation / severity / size) |
| Evaluation axis 1: category accuracy vs gold | Implemented (`scripts/12_test_official.py --gold-json`) |
| Evaluation axis 2: METEOR vs gold descriptions | Implemented (NLTK METEOR) |
| Reproducible offline code | This script + `src/damage_vlm/` |

The dataset contains 110 structural damage images, named 001.jpg to 110.jpg. Participants are required to use the Specified Prompt Template to enable the model to accomplish the following tasks: (1) determine the damage categories present in the image; (2) provide damage characteristic descriptions, including fine-grained information such as crack orientation, corrosion severity, void size.

## Model under test

| Field | Value |
|-------|-------|
| Adapter | `outputs/runs/full_winner_v2` |
| Merged model | `None` |
| Quantization | `none` |
| Predictions | `/home/ai/ethan/Project3/outputs/submissions/official_test.json` |
| Images scored | 110 / 110 |

## Prediction summary (this run)

| Statistic | Value |
|-----------|-------|
| Q1 = yes (has damage) | 110 |
| Q1 = no (no closed-set damage) | 0 |
| Empty descriptions | 0 |
| Mean categories / image | 2.518 |

### Predicted category histogram

| Category | Count |
|----------|-------|
| `spalling` | 79 |
| `exposed_rebar` | 53 |
| `voids` | 43 |
| `cracks` | 41 |
| `corrosion` | 27 |
| `looseness` | 13 |
| `potholes` | 13 |
| `honeycomb` | 5 |
| `efflorescence` | 2 |
| `peeling` | 1 |

## Official scores

The public test package does **not** include hidden ground-truth labels.
Category accuracy and METEOR can be computed only after the organizer gold
file is supplied:

```bash
python scripts/12_test_official.py \
  --predictions-json outputs/submissions/official_test.json \
  --gold-json path/to/official_gold.json
```


## Sample predictions

```json
[
  {
    "image_id": "001",
    "q1_has_structural_damage": true,
    "damage_categories": [
      "cracks"
    ],
    "description": "A crack is observed on the concrete surface, spreading from left to right in an irregular curved shape."
  },
  {
    "image_id": "002",
    "q1_has_structural_damage": true,
    "damage_categories": [
      "cracks"
    ],
    "description": "A crack is observed on the concrete surface, spreading from left to right in an irregular curved shape."
  },
  {
    "image_id": "003",
    "q1_has_structural_damage": true,
    "damage_categories": [
      "cracks"
    ],
    "description": "A crack is observed on the concrete surface, spreading from left to right in an irregular curved shape."
  }
]
```

## How this maps onto the organizer questions

- **Q1** is answered as `q1_has_structural_damage` (true iff at least one closed-set category is predicted) plus the multi-label `damage_categories` list.
- **Q2** is answered as the technical `description` after the shared post-process stack (JSON repair, synonym map, category–text alignment, sentence de-duplication).

## Reproducibility

```bash
python scripts/12_test_official.py --adapter outputs/runs/full_winner_v2
# Low VRAM:
python scripts/12_test_official.py --load-in-4bit --adapter outputs/runs/full_winner_v2
```
