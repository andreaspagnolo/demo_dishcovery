# Dishcovery Demo App

## 1. Overview

This repository contains the browser demo app for the Dishcovery food-image
understanding pipelines on Jetson Orin.

The demo exposes one web UI for:

- Task 1: multi-label ingredient recognition with SigLIP2 candidate recall and
  Qwen3-VL visual selection.
- Task 2: image-to-caption retrieval with SigLIP2 caption recall.
- Calorie estimation with Qwen3-VL dish-composition parsing and a local calorie
  table.
- Browser image input through dataset samples, upload, or camera capture.
- Browser voice commands, optional local Piper TTS, latency/power diagnostics,
  and a personal nutrition history area.

`demo_web_app.py` is the canonical entrypoint. The older OpenCV desktop
webcam/voice entrypoint is intentionally not part of this repository because the
browser app covers the demo workflow.

## 2. Quickstart

Create an environment and install dependencies:

```bash
cd demo_dishcovery
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Jetson Orin, install the NVIDIA/JetPack-compatible PyTorch and torchvision
builds before installing the remaining requirements.

Some Hugging Face models may require authentication. Set one of these before
starting the app:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

## 3. Repository Structure

```text
demo_dishcovery/
|-- demo/
|   |-- command_parser.py
|   |-- history.py
|   |-- nutrition.py
|   |-- stt.py
|   |-- task_router.py
|   `-- web_static/
|       |-- index.html
|       |-- app.js
|       |-- styles.css
|       `-- assets/
|-- dataset/
|   |-- images.txt
|   |-- captions_cleaned.txt
|   |-- captions_cleaned_calories_realistic_average_portions.csv
|   `-- food500_subset/
|       |-- images.txt
|       `-- manifest.csv
|-- labels/
|   |-- MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json
|   |-- evaluation_data.json
|   `-- image_ground_truth_rows.csv
|-- demo_web_app.py
|-- orin_demo.py
|-- orin_task2_demo.py
|-- orin_calorie_demo.py
|-- measurement_utils.py
|-- requirements.txt
`-- README.md
```

## 4. Data Setup and External Resources

The repository commits code, image lists, label files, and metadata. The actual
image folders, downloaded model weights, embedding caches, and generated demo
runs are external and ignored by git.

The default demo expects these local image folders when using dataset samples:

```text
dataset/MM-Food-100K-images-filtered/
dataset/food500_subset/images/
```

The app can still accept uploaded images or camera captures, but the default
sample-image carousel needs `dataset/MM-Food-100K-images-filtered/`.

### 4.1 MM-Food-100K Images

MM-Food-100K is used for Task 1 ingredient recognition and for the default demo
sample images. The Hugging Face dataset repository is `Codatta/MM-Food-100K`.

Install or update `huggingface_hub`, then download the dataset snapshot:

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Codatta/MM-Food-100K",
    repo_type="dataset",
    local_dir="dataset/MM-Food-100K/raw",
    max_workers=50,
)
PY
```

`repo_type="dataset"` is required because `snapshot_download` otherwise treats
the repository as a model repository by default.

The browser demo uses the filtered image folder:

```text
dataset/MM-Food-100K-images-filtered/
```

Populate that folder with the image filenames listed in:

```text
dataset/images.txt
```

If the downloaded snapshot contains URLs or metadata rather than materialized
image files, download the referenced images and keep the filenames aligned with
`dataset/images.txt`.

Required committed metadata for Task 1 and calories:

| File | Purpose |
| --- | --- |
| `dataset/images.txt` | Default Food100K demo sample image list. |
| `dataset/captions_cleaned.txt` | Cleaned ingredient vocabulary. |
| `dataset/captions_cleaned_calories_realistic_average_portions.csv` | Calorie lookup table used by the personal nutrition and calorie-estimation flows. |
| `labels/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json` | Food100K image-to-ingredient metadata. |
| `labels/image_ground_truth_rows.csv` | Food100K image-to-label row map. |

### 4.2 Food500-Cap / ISIA Food-500 Images

Task 2 uses Food500-Cap caption metadata committed in
`labels/evaluation_data.json`. The corresponding images come from the ISIA
Food-500 dataset.

Download the ISIA Food-500 images from the official dataset page:

```text
http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html
```

After downloading and extracting the dataset, copy or symlink the Food500-Cap
subset images used by this repository into:

```text
dataset/food500_subset/images/
```

The subset image order and metadata are defined by:

| File | Purpose |
| --- | --- |
| `dataset/food500_subset/images.txt` | Fixed Food500-Cap subset image list. |
| `dataset/food500_subset/manifest.csv` | Food500-Cap image and class metadata. |
| `labels/evaluation_data.json` | Food500-Cap candidate captions. |

Verify that the local image folder matches the expected image list:

```bash
head dataset/food500_subset/images.txt
ls dataset/food500_subset/images/ | head
```

### 4.3 Expected Final Data Layout

After restoring external image files, the dataset folder should look like this:

```text
dataset/
|-- MM-Food-100K/
|   `-- raw/
|-- MM-Food-100K-images-filtered/
|   |-- img_000000.jpg
|   |-- img_000001.jpg
|   `-- ...
|-- food500_subset/
|   |-- images/
|   |-- images.txt
|   `-- manifest.csv
|-- images.txt
|-- captions_cleaned.txt
`-- captions_cleaned_calories_realistic_average_portions.csv
```

Generated files are recreated at runtime and should not be committed:

```text
demo_runs/
embeddings/
models/
reports/
outputs/
__pycache__/
```

## 5. Run the Web Demo

Start the app:

```bash
python demo_web_app.py
```

Open:

```text
http://127.0.0.1:8787
```

The only runtime inputs exposed by the app are the dataset sample paths:

```bash
python demo_web_app.py \
  --sample-image-dir dataset/MM-Food-100K-images-filtered \
  --sample-images-list dataset/images.txt \
  --sample-image-count 200
```

If you only want to test that the HTTP server starts without loading dataset
samples:

```bash
python demo_web_app.py --sample-image-count 0
```

The app writes run history and nutrition data under:

```text
demo_runs/web/
```

## 6. Optional Local Audio Output

The browser can request server-side TTS through Piper when the executable and
model files are available locally:

```text
models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

If Piper is not installed, the app still runs normally; only spoken audio output
is unavailable.

Browser voice commands use `faster-whisper` on the server to transcribe a WAV
recorded in the browser. Supported command intents are:

```text
find ingredients
describe the dish
estimate calories
execute both
```
