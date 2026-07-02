# Dishcovery Demo App

## 1. Overview

This repository contains the browser demo app for the Dishcovery food-image
understanding pipelines on Jetson Orin.

The demo exposes one web UI for:

- Task 1: multi-label ingredient recognition with SigLIP2 candidate recall and
  Qwen3-VL visual selection.
- Task 2: image-to-caption retrieval with SigLIP2 caption recall and
  Qwen3-VL-Reranker-2B guarded re-ranking for ambiguous images.
- Calorie estimation with Qwen3-VL dish-composition parsing and a local calorie
  table.
- Browser image input through dataset samples, upload, or camera capture.
- Browser voice commands through `faster-whisper`.
- Optional local Piper TTS output.
- Latency, power, CPU/GPU, RAM diagnostics and a personal nutrition history
  area.

`demo_web_app.py` is the canonical entrypoint. The older OpenCV desktop
webcam/voice entrypoint is intentionally not part of this repository because the
browser app covers the demo workflow.

## 2. Tested Jetson Environment

The demo was prepared and tested on this machine:

| Component | Version |
| --- | --- |
| Device | NVIDIA Jetson AGX Orin Developer Kit |
| Architecture | `aarch64` |
| RAM | 64 GB class device, observed as 61 GiB usable RAM |
| Ubuntu | 22.04.5 LTS (`jammy`) |
| Kernel | `5.15.185-tegra` |
| NVIDIA L4T | `R36.5.0` |
| JetPack | `6.2.2+b24` |
| CUDA toolkit | 12.6, `nvcc` release `V12.6.68` |
| Python | 3.10.20 |
| Jetson power mode | `MODE_50W`, mode id `3` |

Useful commands to verify another Jetson:

```bash
tr -d '\0' < /proc/device-tree/model
cat /etc/nv_tegra_release
lsb_release -a
dpkg-query -W nvidia-jetpack nvidia-l4t-core 'cuda-toolkit-*'
nvcc --version
python --version
nvpmodel -q
```

For comparable demo latency, use the same Jetson power mode:

```bash
sudo nvpmodel -m 3
```

`jetson_clocks` is not required for app correctness. If you enable it for stable
benchmark-style latency, record that separately because it changes performance:

```bash
sudo jetson_clocks
sudo jetson_clocks --show
```

The active Python environment used these core package versions:

| Package | Version |
| --- | --- |
| `torch` | 2.8.0 |
| `torchvision` | 0.23.0 |
| `transformers` | 5.9.0 |
| `accelerate` | 1.13.0 |
| `open_clip_torch` | 3.3.0 |
| `qwen-vl-utils` | 0.0.14 |
| `faster-whisper` | 1.2.1 |
| `ctranslate2` | 4.7.2 |
| `av` | 17.1.0 |
| `huggingface_hub` | 1.15.0 |
| `sentence-transformers` | 5.5.1 |
| `piper-tts` | 1.4.2, optional TTS |
| `onnxruntime` | 1.23.2, optional TTS dependency |
| `Pillow` | 12.2.0 |
| `numpy` | 1.26.4 |

## 3. Quickstart

Create an environment and install dependencies:

```bash
cd demo_dishcovery
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Jetson Orin, install the NVIDIA/JetPack-compatible PyTorch and torchvision
builds first if the plain PyPI wheels do not match your JetPack/CUDA setup.
Then install the remaining requirements.

Some Hugging Face models may require authentication. Set one of these before
downloading models or starting the app:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

## 4. Repository Structure

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

## 5. Models

The web demo preloads these default runtime backends:

```text
task1, task2_fast, calories
```

The exact model IDs used by the demo are:

| Demo component | Model |
| --- | --- |
| Task 1 SigLIP2 visual/text recall | `timm/ViT-gopt-16-SigLIP2-384` through OpenCLIP as `hf-hub:timm/ViT-gopt-16-SigLIP2-384` |
| Task 1 Qwen visual selection | `Qwen/Qwen3-VL-4B-Instruct` |
| Task 2 caption recall | `timm/ViT-gopt-16-SigLIP2-384` |
| Task 2 guarded reranker | `Qwen/Qwen3-VL-Reranker-2B` |
| Calorie composition estimation | `Qwen/Qwen3-VL-4B-Instruct` |
| Browser voice commands | `base.en` through `faster-whisper`, cached as `Systran/faster-whisper-base.en` |
| Optional TTS | Piper voice `en_US-lessac-medium` |

Task 2 now defaults to guarded scoring: it keeps the SigLIP2 top result for
easy images and runs `Qwen/Qwen3-VL-Reranker-2B` only for ambiguous images.
This is wired through `demo/task_router.py` with `--final-score-mode
siglip_guarded`.

### 5.1 Download All Runtime Models

The commands below cache Hugging Face models inside `models/huggingface/`.
`models/` is ignored by Git.

Run these exports in every shell where you download models or run the app:

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
```

Download the default runtime models used by the web demo:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="timm/ViT-gopt-16-SigLIP2-384",
    allow_patterns=[
        "open_clip_config.json",
        "open_clip_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
)

snapshot_download(
    repo_id="Systran/faster-whisper-base.en",
)
PY
```

Download the Task 2 reranker used by the default guarded-scoring path:

```bash
hf download Qwen/Qwen3-VL-Reranker-2B \
    --local-dir ./Qwen3-VL-Reranker-2B
```

Install Piper and download the optional voice files used by `demo_web_app.py`:

```bash
python -m pip install "piper-tts==1.4.2"

python - <<'PY'
from huggingface_hub import hf_hub_download

repo_id = "rhasspy/piper-voices"
for filename in [
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
]:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="models/piper",
    )
PY
```

The Piper files should end up at:

```text
models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

If Piper is not installed, the app still runs; only spoken audio output is
unavailable.

### 5.2 Model Cache Check

After download, verify the local model assets:

```bash
find "$HF_HUB_CACHE" -maxdepth 2 -type d | head
test -f models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
test -f models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## 6. Data Setup and External Resources

The repository commits code, image lists, label files, and metadata. The actual
image folders, downloaded model weights, embedding caches, and generated demo
runs are external and ignored by Git.

The default demo expects these local image folders when using dataset samples:

```text
dataset/MM-Food-100K-images-filtered/
dataset/food500_subset/images/
```

The app can still accept uploaded images or camera captures, but the default
sample-image carousel needs `dataset/MM-Food-100K-images-filtered/`.

### 6.1 MM-Food-100K Images

MM-Food-100K is used for Task 1 ingredient recognition and for the default demo
sample images. The Hugging Face dataset repository is `Codatta/MM-Food-100K`.

`requirements.txt` already installs `huggingface_hub`. If you are only running
the dataset download step in a separate environment, install the same pinned
version first:

```bash
python -m pip install "huggingface_hub==1.15.0"
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

### 6.2 Food500-Cap / ISIA Food-500 Images

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

### 6.3 Expected Final Data Layout

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

## 7. Run the Web Demo

Set the model cache variables in the same shell:

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
```

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

The first real run may take longer because SigLIP text embeddings and caption
embeddings are built and cached under:

```text
embeddings/
```

## 8. Voice Commands and Audio Output

Browser voice commands use `faster-whisper` on the server to transcribe a WAV
recorded in the browser. Supported command intents are:

```text
find ingredients
describe the dish
estimate calories
execute both
```

Piper TTS is used only when both the `piper` executable and the Lessac voice
files are available. Without Piper, the browser UI and all model pipelines still
work.

## 9. Reproduction Checklist

Run these checks before starting a public demo:

```bash
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
PY

test -d dataset/MM-Food-100K-images-filtered
test -f dataset/images.txt
test -f labels/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json
test -f labels/evaluation_data.json
test -d "$HF_HUB_CACHE"
python demo_web_app.py --help
```

For a short HTTP startup smoke test:

```bash
timeout 5 python demo_web_app.py --host 127.0.0.1 --port 8791 --quiet --sample-image-count 0
```

Expected output:

```text
Dishcovery web demo ready at http://127.0.0.1:8791
```
