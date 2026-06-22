# Dishcovery Demo App

Browser demo for Dishcovery food-image understanding on Jetson Orin.

The app runs three default pipelines from one web UI:

- Task 1 ingredient recognition with SigLIP2 candidate recall and Qwen-VL selection.
- Task 2 dish-caption selection with SigLIP2 caption recall.
- Calorie estimation with Qwen-VL composition estimation and a local calorie table.

`demo_web_app.py` is the canonical entrypoint. The older OpenCV webcam/voice entrypoint was removed because the browser app already supports dataset samples, camera capture, upload, browser voice commands, TTS, nutrition history, and diagnostics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Jetson Orin, install the JetPack-compatible PyTorch and torchvision builds first, then install the remaining requirements.

Some Qwen models may require Hugging Face authentication. Set one of:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

## Data Layout

The repository commits the metadata needed by the demo, but not image folders, model weights, embedding caches, or generated reports.

Expected local image folders:

```text
dataset/MM-Food-100K-images-filtered/
dataset/food500_subset/images/
```

Committed metadata:

```text
dataset/images.txt
dataset/captions_cleaned.txt
dataset/captions_cleaned_calories_realistic_average_portions.csv
dataset/food500_subset/images.txt
dataset/food500_subset/manifest.csv
labels/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json
labels/evaluation_data.json
labels/image_ground_truth_rows.csv
```

## Run

```bash
python demo_web_app.py
```

Then open:

```text
http://127.0.0.1:8787
```

The only app-level runtime path choices are the dataset sample inputs:

```bash
python demo_web_app.py \
  --sample-image-dir dataset/MM-Food-100K-images-filtered \
  --sample-images-list dataset/images.txt \
  --sample-image-count 200
```

Runtime outputs are written under `demo_runs/web/`. Generated embeddings, reports, downloaded models, and image folders are ignored by Git.
