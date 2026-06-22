#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from PIL import Image

from measurement_utils import PowerMonitor, build_benchmark_summary, print_benchmark_summary, query_nvpmodel


_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent if _THIS_DIR.name == "code" else _THIS_DIR
LABEL_PREFIX = "a plate of food containing "
DEFAULT_IMAGE_DIR = ROOT / "dataset/MM-Food-100K-images-filtered"
DEFAULT_IMAGES_LIST = ROOT / "dataset/images.txt"
DEFAULT_CLEANED_JSON = ROOT / "labels/MM-Food-100K_image_url_ingredients_cleaned_v1_mapped.json"
DEFAULT_GROUND_TRUTH_MAP = ROOT / "labels/image_ground_truth_rows.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports/orin_cuda_demo"
DEFAULT_SIGLIP_MODEL = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DEFAULT_SIGLIP_PRETRAINED = ""
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_CACHE_VERSION = "orin_openclip_siglip2_v1"
MAX_DISH_INGREDIENTS = 10
SERVER_DEFAULT_HOST = "127.0.0.1"
SERVER_DEFAULT_PORT = 8765
SERVER_REQUEST_ARG_FIELDS = (
    "candidate_list_mode",
    "dynamic_candidate_relative_delta",
    "dynamic_candidate_min_k",
    "dynamic_candidate_max_k",
    "top_k",
    "selector",
    "selector_k",
    "selector_threshold",
    "selector_delta",
    "selector_ratio",
    "selector_max_labels",
    "selector_score_mode",
    "visual_weight",
    "qwen_weight",
    "qwen_possible_weight",
    "qwen_reducer",
    "vlm_prompt_mode",
    "count_select_min_count",
    "count_select_max_count",
    "vlm_family",
    "qwen_max_new_tokens",
    "qwen_min_pixels",
    "qwen_max_pixels",
    "skip_vlm_rel_gap_threshold",
    "skip_vlm_visual_selector",
    "skip_vlm_visual_selector_k",
    "skip_vlm_visual_selector_threshold",
    "skip_vlm_visual_selector_delta",
    "skip_vlm_visual_selector_ratio",
    "skip_vlm_visual_selector_max_labels",
    "skip_vlm_visual_selector_score_mode",
    "mock_qwen_json",
)


def hf_hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_has_required_files(snapshot: Path, required_files: tuple[str, ...]) -> bool:
    if not all((snapshot / name).exists() for name in required_files):
        return False

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.exists():
            continue
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_payload.get("weight_map")
            if not isinstance(weight_map, dict):
                return False
            shard_names = {str(name) for name in weight_map.values()}
        except Exception:
            return False
        if not shard_names or not all((snapshot / name).exists() for name in shard_names):
            return False
    return True


def cached_hf_snapshot(repo_id: str, required_files: tuple[str, ...] = ()) -> Path | None:
    cache_dir = hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidates: list[Path] = []
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        try:
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots_dir / revision)
        except OSError:
            pass

    try:
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(snapshots)
    except OSError:
        pass

    seen: set[Path] = set()
    for snapshot in candidates:
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if snapshot.is_dir() and snapshot_has_required_files(snapshot, required_files):
            return snapshot
    return None


def resolve_open_clip_source(model_name: str) -> str:
    if not model_name.startswith("hf-hub:"):
        return model_name
    repo_id = model_name.removeprefix("hf-hub:")
    snapshot = cached_hf_snapshot(repo_id, ("open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json"))
    if snapshot is None:
        return model_name
    return f"local-dir:{snapshot}"


def resolve_transformers_source(model_id: str) -> tuple[str, bool]:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return str(local_path), True
    if "/" not in model_id:
        return model_id, False
    snapshot = cached_hf_snapshot(model_id, ("config.json", "tokenizer.json"))
    if snapshot is None:
        return model_id, False
    return str(snapshot), True


@dataclass
class Candidate:
    rank: int
    label_id: int
    label: str
    visual_score: float
    qwen_confidence: int = 0
    qwen_score: float | None = None
    qwen_possible_score: float | None = None
    fused_score: float | None = None
    selector_score: float | None = None


@dataclass
class DemoRuntime:
    args: argparse.Namespace
    cleaned_rows: list[dict[str, Any]]
    labels: list[str]
    label_to_id: dict[str, int]
    gt_row_map: dict[str, int]
    text_embeddings: np.ndarray
    text_cache: dict[str, Any]
    embedder: "OpenCLIPSigLIP2Embedder"
    qwen: "VisionLanguageScorer | None"
    model_load_timings: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jetson Orin CUDA demo for Food100K ingredient recognition. The pipeline "
            "uses OpenCLIP SigLIP2 for visual candidate retrieval and a vision-language "
            "model for closed-choice visual confidence scoring."
        )
    )
    parser.add_argument("--image", type=Path, default=None, help="Run one image and write a trace JSON.")
    parser.add_argument("--eval-samples", type=int, default=0, help="Evaluate this many dataset images.")
    parser.add_argument("--eval-first", action="store_true", help="Use the first N image-list entries instead of a seeded random sample.")
    parser.add_argument("--serve", action="store_true", help="Load models once and serve inference requests until interrupted.")
    parser.add_argument("--use-server", action="store_true", help="Send this request to an already running local demo server.")
    parser.add_argument("--server-required", action="store_true", help="Fail instead of falling back to local model loading when --use-server cannot connect.")
    parser.add_argument("--shutdown-server", action="store_true", help="Ask a running local demo server to shut down.")
    parser.add_argument("--server-host", default=SERVER_DEFAULT_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_DEFAULT_PORT)
    parser.add_argument(
        "--no-free-memory",
        "--noFreeMemory",
        dest="no_free_memory",
        action="store_true",
        help="After a local run, keep the loaded models alive by starting the local demo server.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--images-list", type=Path, default=DEFAULT_IMAGES_LIST)
    parser.add_argument("--cleaned-json", type=Path, default=DEFAULT_CLEANED_JSON)
    parser.add_argument("--image-ground-truth-map", type=Path, default=DEFAULT_GROUND_TRUTH_MAP)
    parser.add_argument("--captions", type=Path, default=ROOT / "dataset/captions_cleaned.txt")
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-pretrained", default=DEFAULT_SIGLIP_PRETRAINED)
    parser.add_argument(
        "--vlm-model",
        "--qwen-model",
        dest="qwen_model",
        default=DEFAULT_QWEN_MODEL,
        help=(
            "Vision-language model used for close-choice scoring. "
            "Default keeps the original Qwen3 model."
        ),
    )
    parser.add_argument(
        "--vlm-family",
        choices=("auto", "qwen", "generic"),
        default="auto",
        help=(
            "Preprocessing path for the VLM. auto uses the Qwen path for Qwen models "
            "and the generic Transformers image-text-to-text path otherwise."
        ),
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--torch-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--candidate-list-mode",
        choices=("fixed_topk", "dynamic_relative_delta"),
        default="fixed_topk",
        help=(
            "How to build the SigLIP2 candidate list. fixed_topk keeps --top-k labels. "
            "dynamic_relative_delta keeps labels close to the top SigLIP score, clamped by "
            "--dynamic-candidate-min-k and --dynamic-candidate-max-k."
        ),
    )
    parser.add_argument(
        "--dynamic-candidate-relative-delta",
        type=float,
        default=0.5,
        help="For dynamic_relative_delta, keep scores >= top_score - abs(top_score) * this value.",
    )
    parser.add_argument(
        "--dynamic-candidate-min-k",
        type=int,
        default=12,
        help="Minimum candidate count for dynamic_relative_delta. The measured 3%% policy uses 12.",
    )
    parser.add_argument(
        "--dynamic-candidate-max-k",
        type=int,
        default=150,
        help="Maximum candidate count for dynamic_relative_delta. The measured 3%% policy uses 150.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--selector",
        choices=("topk", "threshold", "top_delta", "top_ratio", "threshold_delta", "threshold_ratio"),
        default="threshold",
    )
    parser.add_argument("--selector-k", type=int, default=1)
    parser.add_argument("--selector-threshold", type=float, default=0.92)
    parser.add_argument("--selector-delta", type=float, default=0.9)
    parser.add_argument("--selector-ratio", type=float, default=0.85)
    parser.add_argument("--selector-max-labels", type=int, default=6)
    parser.add_argument("--selector-score-mode", choices=("raw", "row_z", "row_minmax"), default="row_minmax")
    parser.add_argument("--visual-weight", type=float, default=0.60)
    parser.add_argument("--qwen-weight", type=float, default=0.40)
    parser.add_argument(
        "--qwen-possible-weight",
        type=float,
        default=0.0,
        help=(
            "Optional extra fusion weight for VLM confidence=1 candidates. Keep this low; "
            "the confidence=1 branch has high recall but many false positives."
        ),
    )
    parser.add_argument("--qwen-reducer", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--vlm-prompt-mode",
        choices=("confidence", "count_select", "aligned_count_select"),
        default="count_select",
        help=(
            "count_select asks the VLM once to return a visible ingredient count "
            "and selected SigLIP candidate IDs capped by that count. "
            "aligned_count_select counts only visible ingredients represented in the candidate list "
            "and reports unlisted visible ingredients separately. "
            "confidence keeps the original one-pass candidate confidence prompt."
        ),
    )
    parser.add_argument(
        "--count-select-min-count",
        type=int,
        default=1,
        help="Minimum ingredient count accepted from the count_select prompt.",
    )
    parser.add_argument(
        "--count-select-max-count",
        type=int,
        default=0,
        help="Maximum ingredient count accepted from the count_select prompt. 0 means use --top-k.",
    )
    parser.add_argument("--vlm-max-new-tokens", "--qwen-max-new-tokens", dest="qwen_max_new_tokens", type=int, default=48)
    parser.add_argument("--vlm-min-pixels", "--qwen-min-pixels", dest="qwen_min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--vlm-max-pixels", "--qwen-max-pixels", dest="qwen_max_pixels", type=int, default=768 * 28 * 28)
    parser.add_argument(
        "--skip-vlm-rel-gap-threshold",
        type=float,
        default=None,
        help=(
            "Skip VLM close-choice scoring when (top1_visual - top2_visual) / abs(top1_visual) "
            "is at least this value. A measured 350-subset knee point is 0.25."
        ),
    )
    parser.add_argument(
        "--skip-vlm-visual-selector",
        choices=("topk", "threshold", "top_delta", "top_ratio", "threshold_delta", "threshold_ratio"),
        default="top_ratio",
        help="Selector used only for rows where --skip-vlm-rel-gap-threshold fires.",
    )
    parser.add_argument("--skip-vlm-visual-selector-k", type=int, default=1)
    parser.add_argument("--skip-vlm-visual-selector-threshold", type=float, default=0.0)
    parser.add_argument("--skip-vlm-visual-selector-delta", type=float, default=0.0)
    parser.add_argument("--skip-vlm-visual-selector-ratio", type=float, default=0.85)
    parser.add_argument("--skip-vlm-visual-selector-max-labels", type=int, default=3)
    parser.add_argument(
        "--skip-vlm-visual-selector-score-mode",
        choices=("raw", "row_z", "row_minmax"),
        default="raw",
    )
    parser.add_argument("--text-cache", type=Path, default=ROOT / "embeddings/orin_siglip2_text_feats_cache.npz")
    parser.add_argument("--no-text-cache", action="store_true")
    parser.add_argument("--mock-qwen-json", default="", help="Skip Qwen and use this JSON for parser tests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--measure", action="store_true", help="Add benchmark latency, throughput, power, and energy summary to eval output.")
    parser.add_argument("--measure-power", action="store_true", help="Sample Jetson tegrastats power during the per-image inference loop.")
    parser.add_argument("--power-sample-interval-ms", type=int, default=200)
    parser.add_argument("--w-config", default="", help="Power-budget label for the run, for example 15W, 30W, or 50W.")
    parser.add_argument(
        "--moe-expert-usage-json",
        type=Path,
        default=None,
        help="Write MoE router expert-use counts for VLM calls to this JSON file.",
    )
    parser.add_argument(
        "--moe-expert-top-k",
        type=int,
        default=0,
        help="Override router top-k for expert counting. Default uses the model/layer config.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def strip_caption(value: str) -> str:
    text = value.strip()
    if text.lower().startswith(LABEL_PREFIX):
        text = text[len(LABEL_PREFIX) :]
    return normalize_text(text)


def read_cleaned_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a list")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict) or not isinstance(row.get("ingredients"), list):
            raise TypeError(f"Invalid ground-truth row {idx} in {path}")
        rows.append(row)
    return rows


def read_labels(captions: Path, cleaned_rows: list[dict[str, Any]]) -> list[str]:
    if captions.exists():
        labels = [strip_caption(line) for line in captions.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        labels = sorted({normalize_text(item) for row in cleaned_rows for item in row["ingredients"]})
    if not labels:
        raise ValueError("No labels found")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate labels after normalization: {duplicates[:20]}")
    return labels


def image_index(path_or_name: str | Path) -> int:
    match = re.search(r"(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not extract Food100K row index from image name: {path_or_name}")
    return int(match.group(1))


def read_ground_truth_row_map(path: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    if not path.exists():
        return {}
    row_map: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "ground_truth_row"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            image_name = str(row.get("image") or "").strip()
            raw_gt = str(row.get("ground_truth_row") or "").strip()
            if not image_name or not raw_gt:
                continue
            gt_row = int(raw_gt)
            if not 0 <= gt_row < len(rows):
                raise ValueError(f"{path} maps {image_name} to invalid ground-truth row {gt_row}")
            row_map[image_name] = gt_row
    return row_map


def ground_truth_ids_for_image(
    image_name: str,
    rows: list[dict[str, Any]],
    label_to_id: dict[str, int],
    row_map: dict[str, int] | None = None,
) -> list[int]:
    if row_map is not None and image_name in row_map:
        idx = row_map[image_name]
    else:
        idx = image_index(image_name)
    if not 0 <= idx < len(rows):
        map_hint = " via filename index"
        if row_map is not None and image_name in row_map:
            map_hint = " via image-ground-truth map"
        raise IndexError(f"{image_name} maps to ground-truth row {idx}{map_hint}, but only {len(rows)} rows exist")
    ids: list[int] = []
    seen: set[int] = set()
    for ingredient in rows[idx]["ingredients"]:
        label_id = label_to_id.get(normalize_text(ingredient))
        if label_id is None:
            continue
        if label_id not in seen:
            ids.append(label_id)
            seen.add(label_id)
    return ids


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    return normalize_rows(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]


def row_z_1d(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    std = float(scores.std())
    if std < 1e-6:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - float(scores.mean())) / std


def row_minmax_1d(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-6:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def score_view(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.asarray(scores, dtype=np.float32)
    if mode == "row_z":
        return row_z_1d(scores)
    if mode == "row_minmax":
        return row_minmax_1d(scores)
    raise ValueError(f"Unsupported score mode: {mode}")


def sha256_text(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "orin_demo.py requires PyTorch in the dishcovery_andrea environment. "
            "On Jetson Orin, install an NVIDIA/JetPack-compatible PyTorch build, "
            "then add transformers, accelerate, open_clip_torch, qwen-vl-utils, and pillow."
        ) from exc
    return torch


class OpenCLIPSigLIP2Embedder:
    def __init__(self, model_name: str, pretrained: str, device: str, torch_dtype: str) -> None:
        self.torch = require_torch()
        try:
            import open_clip
        except Exception as exc:
            raise RuntimeError("Install open_clip_torch to use the SigLIP2 CUDA embedder.") from exc
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        self.device = self.torch.device(device)
        self.autocast_dtype = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }[torch_dtype]
        precision_arg = {
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
        }[torch_dtype]
        pretrained_arg = pretrained or None
        model_source = resolve_open_clip_source(model_name)
        if model_source != model_name:
            print(f"Using cached OpenCLIP snapshot: {model_source.removeprefix('local-dir:')}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_source,
            pretrained=pretrained_arg,
            precision=precision_arg,
            device=self.device,
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_source)

    def encode_texts(self, prompts: list[str], batch_size: int = 64) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(feats.float().cpu().numpy())
        return normalize_rows(np.concatenate(chunks, axis=0))

    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalize_vector(feats.float().cpu().numpy()[0])


class MoeExpertUsageTracker:
    NUM_EXPERTS_KEYS = ("num_experts", "num_routed_experts", "n_routed_experts")
    TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k")
    CONFIG_CHILD_KEYS = ("text_config", "llm_config", "language_config")

    def __init__(self, model: Any, torch_module: Any, forced_top_k: int | None = None) -> None:
        self.model = model
        self.torch = torch_module
        self.forced_top_k = forced_top_k if forced_top_k and forced_top_k > 0 else None
        self.num_experts = self._find_config_int(self.NUM_EXPERTS_KEYS)
        self.default_top_k = self.forced_top_k or self._find_config_int(self.TOP_K_KEYS) or 1
        self.layer_counts: dict[str, list[int]] = {}
        self.layer_tokens: dict[str, int] = {}
        self.layer_top_k: dict[str, int] = {}
        self.handles: list[Any] = []
        self.warning: str | None = None

    def _config_candidates(self) -> list[Any]:
        root = getattr(self.model, "config", None)
        candidates = [root] if root is not None else []
        for key in self.CONFIG_CHILD_KEYS:
            child = getattr(root, key, None) if root is not None else None
            if child is not None:
                candidates.append(child)
        return candidates

    def _find_config_int(self, keys: tuple[str, ...]) -> int | None:
        for config in self._config_candidates():
            for key in keys:
                value = getattr(config, key, None)
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _module_top_k(self, module_name: str, modules_by_name: dict[str, Any]) -> int:
        if self.forced_top_k:
            return self.forced_top_k
        parent_name = module_name.rsplit(".", 1)[0]
        parent = modules_by_name.get(parent_name)
        if parent is not None:
            for key in self.TOP_K_KEYS:
                value = getattr(parent, key, None)
                if value is None:
                    continue
                try:
                    return max(1, int(value))
                except (TypeError, ValueError):
                    continue
        return max(1, int(self.default_top_k or 1))

    def register(self) -> int:
        modules_by_name = dict(self.model.named_modules())
        candidates: list[tuple[str, Any]] = []
        for name, module in modules_by_name.items():
            if not isinstance(module, self.torch.nn.Linear):
                continue
            if not name.endswith(".gate"):
                continue
            if int(getattr(module, "out_features", 0)) <= 1:
                continue
            lowered = name.lower()
            if "mlp" not in lowered and "moe" not in lowered and "expert" not in lowered:
                continue
            candidates.append((name, module))

        if self.num_experts is None and candidates:
            out_feature_counts: dict[int, int] = {}
            for _, module in candidates:
                out_features = int(module.out_features)
                out_feature_counts[out_features] = out_feature_counts.get(out_features, 0) + 1
            self.num_experts = max(out_feature_counts.items(), key=lambda item: item[1])[0]

        if self.num_experts is None:
            self.warning = "No MoE expert count was found in the model config or router gate modules."
            return 0

        for name, module in candidates:
            if int(module.out_features) != int(self.num_experts):
                continue
            self.layer_counts[name] = [0 for _ in range(int(self.num_experts))]
            self.layer_tokens[name] = 0
            self.layer_top_k[name] = self._module_top_k(name, modules_by_name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

        if not self.handles:
            self.warning = "No MoE router gate modules were found; expert usage was not recorded."
        return len(self.handles)

    def _make_hook(self, layer_name: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                self.record(layer_name, output)
            except Exception as exc:
                if self.warning is None:
                    self.warning = f"MoE expert usage hook failed for {layer_name}: {exc}"

        return hook

    def record(self, layer_name: str, output: Any) -> None:
        if self.num_experts is None:
            return
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(logits, "detach") or int(logits.shape[-1]) != int(self.num_experts):
            return
        flat = logits.detach().reshape(-1, int(self.num_experts))
        if flat.numel() == 0:
            return
        k = min(max(1, int(self.layer_top_k.get(layer_name, self.default_top_k))), int(self.num_experts))
        selected = self.torch.topk(flat.float(), k=k, dim=-1).indices.reshape(-1).to("cpu")
        counts = self.torch.bincount(selected, minlength=int(self.num_experts)).tolist()
        layer_counts = self.layer_counts.setdefault(layer_name, [0 for _ in range(int(self.num_experts))])
        for idx, value in enumerate(counts):
            layer_counts[idx] += int(value)
        self.layer_tokens[layer_name] = self.layer_tokens.get(layer_name, 0) + int(flat.shape[0])

    @staticmethod
    def _ranked_counts(counts: list[int]) -> list[dict[str, Any]]:
        total = int(sum(counts))
        rows = [
            {"expert": idx, "calls": int(value), "fraction": (float(value) / total if total else 0.0)}
            for idx, value in enumerate(counts)
        ]
        return sorted(rows, key=lambda row: (-int(row["calls"]), int(row["expert"])))

    def summary(self, model_id: str) -> dict[str, Any]:
        num_experts = int(self.num_experts or 0)
        aggregate = [0 for _ in range(num_experts)]
        for counts in self.layer_counts.values():
            for idx, value in enumerate(counts):
                aggregate[idx] += int(value)
        layers: list[dict[str, Any]] = []
        for name in sorted(self.layer_counts):
            counts = self.layer_counts[name]
            layers.append(
                {
                    "name": name,
                    "top_k": int(self.layer_top_k.get(name, self.default_top_k)),
                    "routed_token_positions": int(self.layer_tokens.get(name, 0)),
                    "total_expert_calls": int(sum(counts)),
                    "counts_by_expert": [{"expert": idx, "calls": int(value)} for idx, value in enumerate(counts)],
                    "top_experts": self._ranked_counts(counts)[:10],
                    "unused_experts": [idx for idx, value in enumerate(counts) if int(value) == 0],
                }
            )
        return {
            "schema_version": "moe_expert_usage_v1",
            "model_id": model_id,
            "enabled": True,
            "registered_gate_count": len(self.handles),
            "registered_gate_modules": sorted(self.layer_counts),
            "num_experts": num_experts,
            "default_top_k": int(self.default_top_k),
            "forced_top_k": self.forced_top_k,
            "warning": self.warning,
            "routed_token_positions": int(sum(self.layer_tokens.values())),
            "total_expert_calls": int(sum(aggregate)),
            "aggregate_counts_by_expert": [{"expert": idx, "calls": int(value)} for idx, value in enumerate(aggregate)],
            "aggregate_top_experts": self._ranked_counts(aggregate),
            "aggregate_unused_experts": [idx for idx, value in enumerate(aggregate) if int(value) == 0],
            "layers": layers,
        }


def load_or_build_text_embeddings(
    embedder: OpenCLIPSigLIP2Embedder,
    labels: list[str],
    model_name: str,
    pretrained: str,
    cache_path: Path,
    use_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    label_hash = sha256_text(labels)
    cache_info = {"enabled": bool(use_cache), "path": str(cache_path), "hit": False, "label_hash": label_hash}
    if use_cache and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if (
                str(data["cache_version"]) == TEXT_CACHE_VERSION
                and str(data["label_hash"]) == label_hash
                and str(data["model_name"]) == model_name
                and str(data["pretrained"]) == pretrained
            ):
                cache_info["hit"] = True
                return normalize_rows(data["text_embeddings"]), cache_info
        except Exception as exc:
            cache_info["load_warning"] = str(exc)
    prompts = [f"{LABEL_PREFIX}{label}" for label in labels]
    text_embeddings = embedder.encode_texts(prompts)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            text_embeddings=text_embeddings.astype(np.float32),
            labels=np.asarray(labels),
            label_hash=np.asarray(label_hash),
            model_name=np.asarray(model_name),
            pretrained=np.asarray(pretrained),
            cache_version=np.asarray(TEXT_CACHE_VERSION),
        )
    return text_embeddings, cache_info


def build_candidates(labels: list[str], visual_scores: np.ndarray, top_k: int) -> list[Candidate]:
    order = np.argsort(-visual_scores)[: min(top_k, len(labels))]
    return [
        Candidate(rank=rank, label_id=int(label_id), label=labels[int(label_id)], visual_score=float(visual_scores[int(label_id)]))
        for rank, label_id in enumerate(order, start=1)
    ]


def candidate_count_for_scores(visual_scores: np.ndarray, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    scores = np.asarray(visual_scores, dtype=np.float32)
    label_count = int(scores.shape[0])
    if label_count <= 0:
        raise ValueError("No labels available for candidate selection")

    mode = str(args.candidate_list_mode)
    if mode == "fixed_topk":
        count = min(max(1, int(args.top_k)), label_count)
        return count, {
            "mode": mode,
            "top_k": int(args.top_k),
            "effective_k": count,
        }

    if mode == "dynamic_relative_delta":
        relative_delta = float(args.dynamic_candidate_relative_delta)
        if relative_delta < 0.0:
            raise ValueError("--dynamic-candidate-relative-delta must be non-negative")
        min_k = max(1, int(args.dynamic_candidate_min_k))
        raw_max_k = int(args.dynamic_candidate_max_k)
        max_k = label_count if raw_max_k <= 0 else min(max(1, raw_max_k), label_count)
        if min_k > max_k:
            raise ValueError("--dynamic-candidate-min-k must be <= --dynamic-candidate-max-k")
        top_score = float(scores.max())
        threshold = top_score - abs(top_score) * relative_delta
        raw_count = int((scores >= threshold).sum())
        count = min(max(raw_count, min_k), max_k)
        return count, {
            "mode": mode,
            "relative_delta": relative_delta,
            "min_k": min_k,
            "max_k": max_k,
            "top_score": top_score,
            "score_threshold": threshold,
            "raw_dynamic_k": raw_count,
            "effective_k": count,
        }

    raise ValueError(f"Unsupported candidate list mode: {mode}")


def build_qwen_prompt(candidates: list[Candidate]) -> str:
    lines = "\n".join(f"{candidate.rank}. {candidate.label}" for candidate in candidates)
    return (
        "Classify each candidate ingredient by direct visual evidence in this food image.\n\n"
        "Confidence scale:\n"
        "2 = clearly visible as a physical food item or ingredient\n"
        "1 = possibly visible, partially hidden, ambiguous, or weakly supported\n"
        "0 = not visible\n\n"
        "Rules:\n"
        "- Use only the numbered candidate IDs below.\n"
        "- Do not invent labels or use synonyms.\n"
        "- Do not infer recipe ingredients, sauces, seasonings, or cuisine context unless visible.\n"
        "- Return every candidate ID exactly once.\n"
        "- Return only valid JSON. No markdown and no explanation.\n\n"
        'Output format example: {"1":2,"2":0,"3":1}\n\n'
        "Candidate ingredients:\n"
        f"{lines}"
    )


def build_count_select_prompt(candidates: list[Candidate], min_count: int, max_count: int) -> str:
    lines = "\n".join(f"{candidate.rank}. {candidate.label}" for candidate in candidates)
    return (
        "Analyze the food image using only the candidate ingredient list.\n\n"
        "Rules:\n"
        "- Select only numbered candidate IDs that are clearly visible food ingredients.\n"
        "- Do not infer recipe ingredients, sauces, seasonings, or cuisine context unless visible.\n"
        "- Count visible physical ingredient types, not pieces or hidden recipe components.\n"
        f"- The count must be from {min_count} to {max_count}.\n"
        "- If fewer candidate labels are clearly visible than the count, return only those visible IDs.\n"
        "- Return only one compact JSON array. No markdown, keys, or explanation.\n"
        "- Format: [count,[candidate_id,...]]. Example: [3,[1,4,7]].\n\n"
        "Candidate ingredients:\n"
        f"{lines}"
    )


def build_aligned_count_select_prompt(candidates: list[Candidate], min_count: int, max_count: int) -> str:
    lines = "\n".join(f"{candidate.rank}. {candidate.label}" for candidate in candidates)
    return (
        "Analyze the food image using only the candidate ingredient list as the allowed label space.\n\n"
        "Task:\n"
        "1. Select candidate IDs whose labels are clearly visible in the image.\n"
        "2. Set count to the number of selected candidate IDs.\n"
        "3. If a visible ingredient is not represented by any candidate label, put a short name in unlisted_visible.\n\n"
        "Rules:\n"
        "- Use only numbered candidate IDs in selected.\n"
        "- Do not select a candidate just because it is visually similar to an unlisted ingredient.\n"
        "- Do not infer hidden recipe ingredients, seasonings, oils, or sauces unless visibly distinct.\n"
        "- Select a candidate only when it is an exact visual match or a strong food synonym for a visible item.\n"
        "- Visible ingredients absent from the candidate list must go only in unlisted_visible.\n"
        "- Do not count unlisted_visible items.\n"
        "- count must equal the length of selected.\n"
        f"- count must be from {min_count} to {max_count}; if no candidate is clearly visible, select the strongest visible candidate.\n"
        "- Return only one compact JSON array. No markdown, keys, or explanation.\n"
        "- Format: [count,[candidate_id,...],[unlisted_visible,...]]. Example: [2,[1,4],[\"cilantro\"]].\n\n"
        'Schema: {"count": <len(selected)>, "selected": [<candidate_id>, ...], "unlisted_visible": [<name>, ...]}\n\n'
        "Candidate ingredients:\n"
        f"{lines}"
    )


def resolve_vlm_family(model_id: str, family: str) -> str:
    if family != "auto":
        return family
    return "qwen" if "qwen" in model_id.lower() else "generic"


class VisionLanguageScorer:
    def __init__(
        self,
        model_id: str,
        family: str,
        device: str,
        torch_dtype: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
        track_moe_experts: bool = False,
        moe_expert_top_k: int = 0,
    ) -> None:
        self.torch = require_torch()
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        self.device = device
        self.model_id = model_id
        self.family = resolve_vlm_family(model_id, family)
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.moe_tracker: MoeExpertUsageTracker | None = None
        dtype = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }[torch_dtype]
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "VLM inference requires transformers and accelerate. LiquidAI/LFM2.5-VL-1.6B "
                "requires Transformers v5.1 or newer; Qwen3-VL also requires qwen-vl-utils."
            ) from exc
        self.process_vision_info = None
        if self.family == "qwen":
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as exc:
                raise RuntimeError(
                    "Qwen3-VL inference requires transformers, accelerate, and qwen-vl-utils. "
                    "Use recent Transformers with Qwen3-VL support."
                ) from exc
            self.process_vision_info = process_vision_info
        elif self.family != "generic":
            raise ValueError(f"Unsupported VLM family: {self.family}")

        model_source, local_files_only = resolve_transformers_source(model_id)
        if model_source != model_id:
            print(f"Using cached Transformers snapshot: {model_source}")

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.family == "qwen":
            processor_kwargs.update({"min_pixels": min_pixels, "max_pixels": max_pixels})
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
        }
        if local_files_only:
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True
        self.processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs).eval()
        except TypeError:
            if "torch_dtype" in model_kwargs:
                model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs).eval()
        if device == "cpu":
            self.model.to("cpu")
        if track_moe_experts:
            self.moe_tracker = MoeExpertUsageTracker(
                self.model,
                self.torch,
                forced_top_k=moe_expert_top_k if moe_expert_top_k > 0 else None,
            )
            gate_count = self.moe_tracker.register()
            if gate_count:
                print(f"MoE expert usage tracking enabled for {gate_count} router gate modules.")
            else:
                print(f"MoE expert usage tracking warning: {self.moe_tracker.warning}")

    def score(
        self,
        image_path: Path,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> str:
        if self.family == "qwen":
            return self._score_qwen(
                image_path,
                prompt,
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        return self._score_generic(image_path, prompt, max_new_tokens=max_new_tokens)

    def _score_qwen(
        self,
        image_path: Path,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> str:
        if self.process_vision_info is None:
            raise RuntimeError("Qwen preprocessing was not initialized")
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        effective_min_pixels = self.min_pixels if min_pixels is None else int(min_pixels)
        effective_max_pixels = self.max_pixels if max_pixels is None else int(max_pixels)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                        "min_pixels": effective_min_pixels,
                        "max_pixels": effective_max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=effective_max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _score_generic(self, image_path: Path, prompt: str, *, max_new_tokens: int | None = None) -> str:
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=effective_max_new_tokens, do_sample=False)
        input_len = int(inputs["input_ids"].shape[-1])
        trimmed = generated[:, input_len:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def extract_json_object(text: str) -> tuple[Any | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"
    try:
        return json.loads(stripped), None
    except Exception:
        pass
    starts = [idx for idx, char in enumerate(stripped) if char in "{["]
    if not starts:
        return None, "no_json_payload"
    last_error = ""
    for start in starts:
        opener = stripped[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : idx + 1]), None
                    except Exception as exc:
                        last_error = f"invalid_json_payload: {exc}"
                        break
    if last_error:
        return None, last_error
    return None, "unterminated_json_payload"


def coerce_confidence_value(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("confidence", "score", "level", "value"):
            if key in value:
                return coerce_confidence_value(value[key])
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        raw = int(value)
        return raw if raw in (0, 1, 2) else None
    if isinstance(value, str):
        match = re.search(r"[012]", value)
        return int(match.group(0)) if match else None
    return None


def coerce_count_value(value: Any) -> int | None:
    if isinstance(value, list):
        if value:
            return coerce_count_value(value[0])
        return None
    if isinstance(value, dict):
        for key in ("count", "ingredient_count", "visible_ingredient_count", "num_ingredients", "n"):
            if key in value:
                return coerce_count_value(value[key])
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        return int(match.group(0)) if match else None
    return None


def parse_visible_ingredient_count(raw_text: str, min_count: int, max_count: int) -> tuple[int, list[str], Any | None]:
    parsed, warning = extract_json_object(raw_text)
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
    count = coerce_count_value(parsed) if parsed is not None else None
    if count is None:
        match = re.search(r"\d+", raw_text)
        if match:
            count = int(match.group(0))
        else:
            count = min_count
            warnings.append("count_parse_failed_used_min_count")
    if count < min_count:
        warnings.append(f"count_clamped_min_{count}_to_{min_count}")
        count = min_count
    if count > max_count:
        warnings.append(f"count_clamped_max_{count}_to_{max_count}")
        count = max_count
    return count, warnings, parsed


def coerce_candidate_rank(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("id", "candidate_id", "candidate", "rank", "number", "value"):
            if key in value:
                return coerce_candidate_rank(value[key])
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        return int(match.group(0)) if match else None
    return None


def parse_selected_candidate_ranks(raw_text: str, candidate_count: int, limit: int) -> tuple[list[int], list[str], Any | None]:
    parsed, warning = extract_json_object(raw_text)
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
    if parsed is None:
        return [], warnings, None

    payload = parsed
    if isinstance(payload, list) and payload and coerce_count_value(payload[0]) is not None:
        if len(payload) >= 2 and isinstance(payload[1], list):
            payload = payload[1]
        else:
            payload = payload[1:]
    if isinstance(payload, dict):
        for key in ("selected", "present", "ids", "ingredients", "candidates", "labels"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    raw_ranks: list[int] = []
    if isinstance(payload, list):
        for item in payload:
            rank = coerce_candidate_rank(item)
            if rank is not None:
                raw_ranks.append(rank)
    elif isinstance(payload, dict):
        for raw_key, raw_value in payload.items():
            key_text = str(raw_key).strip()
            if not key_text.isdigit():
                continue
            confidence = coerce_confidence_value(raw_value)
            include = bool(raw_value) if confidence is None else confidence > 0
            if include:
                raw_ranks.append(int(key_text))
    else:
        warnings.append(f"unsupported_selected_payload_{type(payload).__name__}")

    selected: list[int] = []
    seen: set[int] = set()
    for rank in raw_ranks:
        if not 1 <= rank <= candidate_count:
            warnings.append(f"ignored_out_of_range_selected_id_{rank}")
            continue
        if rank in seen:
            continue
        selected.append(rank)
        seen.add(rank)
        if len(selected) >= limit:
            break
    return selected, warnings, parsed


def parse_unlisted_visible(parsed: Any) -> list[str]:
    if isinstance(parsed, list):
        payload = parsed[2] if len(parsed) >= 3 else None
        if not isinstance(payload, list):
            return []
        values: list[str] = []
        for item in payload:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    if not isinstance(parsed, dict):
        return []
    payload = parsed.get("unlisted_visible")
    if payload is None:
        payload = parsed.get("unlisted")
    if payload is None:
        payload = parsed.get("not_in_list")
    if not isinstance(payload, list):
        return []
    values: list[str] = []
    for item in payload:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def parse_qwen_confidences(raw_text: str, candidate_count: int) -> tuple[dict[int, int], list[str], Any | None]:
    parsed, warning = extract_json_object(raw_text)
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
    confidences = {idx: 0 for idx in range(1, candidate_count + 1)}
    if parsed is None:
        return confidences, warnings, None
    payload = parsed
    if isinstance(payload, dict):
        for key in ("scores", "confidences", "confidence", "ingredients", "candidates"):
            if isinstance(payload.get(key), (dict, list)):
                payload = payload[key]
                break
    if isinstance(payload, list):
        for idx, item in enumerate(payload, start=1):
            if idx <= candidate_count:
                value = coerce_confidence_value(item)
                if value is not None:
                    confidences[idx] = value
        return confidences, warnings, parsed
    if not isinstance(payload, dict):
        warnings.append(f"unsupported_json_payload_{type(payload).__name__}")
        return confidences, warnings, parsed
    for raw_key, raw_value in payload.items():
        key_text = str(raw_key).strip()
        if not key_text.isdigit():
            warnings.append(f"ignored_non_numeric_key_{key_text}")
            continue
        idx = int(key_text)
        if not 1 <= idx <= candidate_count:
            warnings.append(f"ignored_out_of_range_key_{idx}")
            continue
        value = coerce_confidence_value(raw_value)
        if value is None:
            warnings.append(f"invalid_confidence_for_{idx}")
            continue
        confidences[idx] = value
    return confidences, warnings, parsed


def select_final_candidates(
    candidates: list[Candidate],
    selector_scores: np.ndarray,
    selector: str,
    k: int,
    threshold: float,
    delta: float,
    ratio: float,
    max_labels: int,
) -> list[int]:
    candidate_positions = np.asarray([candidate.label_id for candidate in candidates], dtype=np.int64)
    ordered = candidate_positions[np.argsort(-selector_scores[candidate_positions])]
    vals = selector_scores[ordered]
    rank_cap = np.arange(len(vals)) < min(max_labels, len(vals))
    if selector == "topk":
        mask = np.arange(len(vals)) < min(k, len(vals))
    elif selector == "threshold":
        mask = (vals >= threshold) & rank_cap
        mask[0] = True
    elif selector == "top_delta":
        mask = (vals >= vals[0] - delta) & rank_cap
        mask[0] = True
    elif selector == "top_ratio":
        mask = (vals >= vals[0] * ratio) & rank_cap
        mask[0] = True
    elif selector == "threshold_delta":
        mask = (vals >= threshold) & (vals >= vals[0] - delta) & rank_cap
        mask[0] = True
    elif selector == "threshold_ratio":
        mask = (vals >= threshold) & (vals >= vals[0] * ratio) & rank_cap
        mask[0] = True
    else:
        raise ValueError(f"Unsupported selector: {selector}")
    return [int(label_id) for label_id in ordered[mask]]


def qwen_branch_scores(
    label_ids: list[int],
    text_embeddings: np.ndarray,
    reducer: str,
) -> np.ndarray:
    if not label_ids:
        return np.zeros(text_embeddings.shape[0], dtype=np.float32)
    qwen_items = normalize_rows(text_embeddings[label_ids])
    if reducer == "max":
        return np.max(100.0 * (qwen_items @ text_embeddings.T), axis=0)
    if reducer == "mean":
        qwen_embedding = normalize_vector(qwen_items.mean(axis=0))
        return 100.0 * (qwen_embedding @ text_embeddings.T)
    raise ValueError(f"Unsupported Qwen reducer: {reducer}")


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "label_id": candidate.label_id,
        "label": candidate.label,
        "visual_score": candidate.visual_score,
        "qwen_confidence": candidate.qwen_confidence,
        "qwen_score": candidate.qwen_score,
        "qwen_possible_score": candidate.qwen_possible_score,
        "fused_score": candidate.fused_score,
        "selector_score": candidate.selector_score,
    }


def clip_candidate_ranks_by_siglip(selected_ranks: list[int], max_items: int) -> list[int]:
    if max_items <= 0 or len(selected_ranks) <= max_items:
        return list(selected_ranks)
    unique_ranks = sorted({int(rank) for rank in selected_ranks})
    return unique_ranks[:max_items]


def clip_selected_ids_by_siglip(candidates: list[Candidate], selected_ids: list[int], max_items: int) -> list[int]:
    if max_items <= 0 or len(selected_ids) <= max_items:
        return list(selected_ids)
    rank_by_label_id = {candidate.label_id: candidate.rank for candidate in candidates}
    ordered = sorted({int(label_id) for label_id in selected_ids}, key=lambda label_id: rank_by_label_id.get(label_id, 10**9))
    return ordered[:max_items]


def run_one(
    image_path: Path,
    labels: list[str],
    text_embeddings: np.ndarray,
    embedder: OpenCLIPSigLIP2Embedder,
    qwen: VisionLanguageScorer | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if image_path.is_dir():
        raise IsADirectoryError(f"{image_path} is a directory; pass a full image path such as {image_path / 'img_000005.jpg'}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()
    t0 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    image_embedding = embedder.encode_image(image)
    visual_scores = 100.0 * (image_embedding @ text_embeddings.T)
    candidate_count, candidate_list_policy = candidate_count_for_scores(visual_scores, args)
    candidates = build_candidates(labels, visual_scores, candidate_count)
    timings["siglip2_image_and_topk_sec"] = time.perf_counter() - t0

    top1_visual_score = float(candidates[0].visual_score) if candidates else 0.0
    top2_visual_score = float(candidates[1].visual_score) if len(candidates) > 1 else top1_visual_score
    visual_gap = top1_visual_score - top2_visual_score
    visual_rel_gap = visual_gap / max(abs(top1_visual_score), 1e-9)
    skip_vlm_threshold = args.skip_vlm_rel_gap_threshold
    skip_vlm = skip_vlm_threshold is not None and visual_rel_gap >= float(skip_vlm_threshold)

    prompt_mode = str(args.vlm_prompt_mode)
    prompt = build_qwen_prompt(candidates)
    raw_count = ""
    parsed_count = None
    visible_count = None
    selected_candidate_ranks: list[int] = []
    unlisted_visible: list[str] = []
    t0 = time.perf_counter()
    if skip_vlm:
        raw_qwen = "{}"
        parsed_qwen = None
        parse_warnings: list[str] = []
        confidences = {idx: 0 for idx in range(1, len(candidates) + 1)}
    elif prompt_mode == "confidence":
        raw_qwen = args.mock_qwen_json if args.mock_qwen_json else qwen.score(image_path, prompt)  # type: ignore[union-attr]
        confidences, parse_warnings, parsed_qwen = parse_qwen_confidences(raw_qwen, len(candidates))
    elif prompt_mode in ("count_select", "aligned_count_select"):
        min_count = min(max(0, int(args.count_select_min_count)), MAX_DISH_INGREDIENTS)
        max_count = int(args.count_select_max_count) if int(args.count_select_max_count) > 0 else len(candidates)
        max_count = max(min_count, min(max_count, len(candidates), MAX_DISH_INGREDIENTS))
        if prompt_mode == "aligned_count_select":
            prompt = build_aligned_count_select_prompt(candidates, min_count, max_count)
        else:
            prompt = build_count_select_prompt(candidates, min_count, max_count)
        raw_qwen = args.mock_qwen_json if args.mock_qwen_json else qwen.score(image_path, prompt)  # type: ignore[union-attr]
        raw_count = raw_qwen
        visible_count, count_warnings, parsed_count = parse_visible_ingredient_count(raw_qwen, min_count, max_count)
        selected_candidate_ranks, select_warnings, parsed_qwen = parse_selected_candidate_ranks(
            raw_qwen,
            len(candidates),
            visible_count,
        )
        unlisted_visible = parse_unlisted_visible(parsed_qwen)
        parse_warnings = [f"count:{warning}" for warning in count_warnings]
        parse_warnings.extend(f"select:{warning}" for warning in select_warnings)
        if visible_count > 0 and not selected_candidate_ranks:
            selected_candidate_ranks = list(range(1, min(visible_count, len(candidates), MAX_DISH_INGREDIENTS) + 1))
            parse_warnings.append("select_empty_used_top_visual_candidates")
        clipped_candidate_ranks = clip_candidate_ranks_by_siglip(selected_candidate_ranks, MAX_DISH_INGREDIENTS)
        if len(clipped_candidate_ranks) != len(selected_candidate_ranks):
            parse_warnings.append(
                f"select_clipped_to_top_siglip_{MAX_DISH_INGREDIENTS}_from_{len(selected_candidate_ranks)}"
            )
            selected_candidate_ranks = clipped_candidate_ranks
        if visible_count is not None and visible_count > MAX_DISH_INGREDIENTS:
            parse_warnings.append(f"count_clipped_to_{MAX_DISH_INGREDIENTS}_from_{visible_count}")
            visible_count = MAX_DISH_INGREDIENTS
        if prompt_mode == "aligned_count_select":
            aligned_count = max(min_count, min(max_count, len(selected_candidate_ranks)))
            if visible_count != aligned_count:
                parse_warnings.append(f"aligned_count_corrected_{visible_count}_to_{aligned_count}")
                visible_count = aligned_count
        selected_rank_set = set(selected_candidate_ranks)
        confidences = {idx: (2 if idx in selected_rank_set else 0) for idx in range(1, len(candidates) + 1)}
    else:
        raise ValueError(f"Unsupported VLM prompt mode: {prompt_mode}")
    timings["qwen_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for candidate in candidates:
        candidate.qwen_confidence = int(confidences.get(candidate.rank, 0))

    confidence2_label_ids = [candidate.label_id for candidate in candidates if candidate.qwen_confidence == 2]
    confidence1_label_ids = [candidate.label_id for candidate in candidates if candidate.qwen_confidence == 1]
    qwen_scores = qwen_branch_scores(confidence2_label_ids, text_embeddings, args.qwen_reducer)
    qwen_possible_scores = qwen_branch_scores(confidence1_label_ids, text_embeddings, args.qwen_reducer)

    if skip_vlm:
        fused_scores = np.asarray(visual_scores, dtype=np.float32)
        fusion_mode = "visual_only_skip_vlm_rel_gap"
        selector_scores = score_view(fused_scores, args.skip_vlm_visual_selector_score_mode)
        selected_ids = select_final_candidates(
            candidates,
            selector_scores,
            selector=args.skip_vlm_visual_selector,
            k=args.skip_vlm_visual_selector_k,
            threshold=args.skip_vlm_visual_selector_threshold,
            delta=args.skip_vlm_visual_selector_delta,
            ratio=args.skip_vlm_visual_selector_ratio,
            max_labels=args.skip_vlm_visual_selector_max_labels,
        )
        selected_ids = clip_selected_ids_by_siglip(candidates, selected_ids, MAX_DISH_INGREDIENTS)
    elif prompt_mode in ("count_select", "aligned_count_select"):
        fused_scores = args.visual_weight * row_z_1d(visual_scores)
        if confidence2_label_ids and args.qwen_weight:
            fused_scores = fused_scores + args.qwen_weight * row_z_1d(qwen_scores)
        selector_scores = score_view(fused_scores, args.selector_score_mode)
        rank_to_label_id = {candidate.rank: candidate.label_id for candidate in candidates}
        selected_ids = [rank_to_label_id[rank] for rank in selected_candidate_ranks if rank in rank_to_label_id]
        selected_ids = clip_selected_ids_by_siglip(candidates, selected_ids, MAX_DISH_INGREDIENTS)
        fusion_mode = f"vlm_{prompt_mode}_direct_count_{visible_count}"
    else:
        fused_scores = args.visual_weight * row_z_1d(visual_scores)
        fusion_parts = [f"visual_w{args.visual_weight:.3g}"]
        used_qwen_branch = False
        if confidence2_label_ids and args.qwen_weight:
            fused_scores = fused_scores + args.qwen_weight * row_z_1d(qwen_scores)
            fusion_parts.append(f"qwen_confidence2_{args.qwen_reducer}_w{args.qwen_weight:.3g}")
            used_qwen_branch = True
        if confidence1_label_ids and args.qwen_possible_weight:
            fused_scores = fused_scores + args.qwen_possible_weight * row_z_1d(qwen_possible_scores)
            fusion_parts.append(f"qwen_confidence1_{args.qwen_reducer}_w{args.qwen_possible_weight:.3g}")
            used_qwen_branch = True
        if not used_qwen_branch:
            fused_scores = row_z_1d(visual_scores)
            fusion_parts = ["visual_only_no_qwen_branch"]
        fusion_mode = "+".join(fusion_parts)

        selector_scores = score_view(fused_scores, args.selector_score_mode)
        selected_ids = select_final_candidates(
            candidates,
            selector_scores,
            selector=args.selector,
            k=args.selector_k,
            threshold=args.selector_threshold,
            delta=args.selector_delta,
            ratio=args.selector_ratio,
            max_labels=args.selector_max_labels,
        )
        selected_ids = clip_selected_ids_by_siglip(candidates, selected_ids, MAX_DISH_INGREDIENTS)
    for candidate in candidates:
        candidate.qwen_score = float(qwen_scores[candidate.label_id])
        candidate.qwen_possible_score = float(qwen_possible_scores[candidate.label_id])
        candidate.fused_score = float(fused_scores[candidate.label_id])
        candidate.selector_score = float(selector_scores[candidate.label_id])
    timings["fusion_and_selection_sec"] = time.perf_counter() - t0
    timings["total_image_sec"] = time.perf_counter() - total_t0

    return {
        "schema_version": "orin_cuda_siglip2_qwen3vl_v1",
        "image": str(image_path),
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            "qwen_model": args.qwen_model,
            "vlm_model": args.qwen_model,
            "vlm_family": resolve_vlm_family(args.qwen_model, args.vlm_family),
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "candidate_list_mode": args.candidate_list_mode,
            "dynamic_candidate_relative_delta": args.dynamic_candidate_relative_delta,
            "dynamic_candidate_min_k": args.dynamic_candidate_min_k,
            "dynamic_candidate_max_k": args.dynamic_candidate_max_k,
            "max_dish_ingredients": MAX_DISH_INGREDIENTS,
            "effective_candidate_k": len(candidates),
            "top_k": args.top_k,
            "visual_weight": args.visual_weight,
            "qwen_weight": args.qwen_weight,
            "qwen_possible_weight": args.qwen_possible_weight,
            "qwen_reducer": args.qwen_reducer,
            "vlm_prompt_mode": args.vlm_prompt_mode,
            "count_select_min_count": args.count_select_min_count,
            "count_select_max_count": args.count_select_max_count,
            "vlm_max_new_tokens": args.qwen_max_new_tokens,
            "skip_vlm_rel_gap_threshold": args.skip_vlm_rel_gap_threshold,
            "skip_vlm_visual_selector": args.skip_vlm_visual_selector,
            "skip_vlm_visual_selector_score_mode": args.skip_vlm_visual_selector_score_mode,
            "skip_vlm_visual_selector_k": args.skip_vlm_visual_selector_k,
            "skip_vlm_visual_selector_threshold": args.skip_vlm_visual_selector_threshold,
            "skip_vlm_visual_selector_delta": args.skip_vlm_visual_selector_delta,
            "skip_vlm_visual_selector_ratio": args.skip_vlm_visual_selector_ratio,
            "skip_vlm_visual_selector_max_labels": args.skip_vlm_visual_selector_max_labels,
            "selector": args.selector,
            "selector_score_mode": args.selector_score_mode,
            "selector_k": args.selector_k,
            "selector_threshold": args.selector_threshold,
            "selector_delta": args.selector_delta,
            "selector_ratio": args.selector_ratio,
            "selector_max_labels": args.selector_max_labels,
        },
        "candidate_list_policy": candidate_list_policy,
        "fusion_mode": fusion_mode,
        "siglip2_top_candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "qwen": {
            "prompt_mode": prompt_mode,
            "prompt": prompt,
            "count_raw_text": raw_count,
            "count_parsed": parsed_count,
            "visible_ingredient_count": visible_count,
            "selected_candidate_ranks": selected_candidate_ranks,
            "unlisted_visible": unlisted_visible,
            "raw_text": raw_qwen,
            "parsed": parsed_qwen,
            "parse_warnings": parse_warnings,
            "skipped": skip_vlm,
            "skip_reason": "visual_rel_gap" if skip_vlm else None,
        },
        "skip_vlm": {
            "enabled": skip_vlm_threshold is not None,
            "skipped": skip_vlm,
            "rel_gap_threshold": skip_vlm_threshold,
            "top1_visual_score": top1_visual_score,
            "top2_visual_score": top2_visual_score,
            "visual_gap": visual_gap,
            "visual_rel_gap": visual_rel_gap,
        },
        "selected_label_ids": selected_ids,
        "selected_labels": [labels[label_id] for label_id in selected_ids],
        "timings_sec": timings,
    }


def metrics_from_sets(predictions: list[set[int]], truths: list[set[int]]) -> dict[str, Any]:
    tp = fp = fn = exact = 0
    row_f1_sum = 0.0
    for pred, truth in zip(predictions, truths):
        row_tp = len(pred & truth)
        row_fp = len(pred - truth)
        row_fn = len(truth - pred)
        tp += row_tp
        fp += row_fp
        fn += row_fn
        exact += int(pred == truth)
        denom = 2 * row_tp + row_fp + row_fn
        row_f1_sum += (2 * row_tp / denom) if denom else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows = len(predictions)
    return {
        "rows": rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rowavg_f1": row_f1_sum / rows if rows else 0.0,
        "exact_match_rows": exact,
        "mean_predicted_labels": sum(len(p) for p in predictions) / rows if rows else 0.0,
        "mean_ground_truth_labels": sum(len(t) for t in truths) / rows if rows else 0.0,
    }


def read_image_names(images_list: Path, image_dir: Path) -> list[str]:
    if images_list.exists():
        names = [line.strip() for line in images_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = sorted(
            [path.name for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=image_sort_key,
        )
    missing = [name for name in names if not (image_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image-list entries are missing from {image_dir}; examples: {missing[:10]}")
    return names


def image_sort_key(name: str) -> tuple[int, str]:
    try:
        return image_index(name), name
    except ValueError:
        return 10**12, name


def list_eval_images(image_dir: Path, images_list: Path, sample_count: int, seed: int, use_first: bool) -> list[Path]:
    names = read_image_names(images_list, image_dir)
    if sample_count <= 0 or sample_count >= len(names):
        selected_names = names
    elif use_first:
        selected_names = names[:sample_count]
    else:
        rng = random.Random(seed)
        selected_names = sorted(rng.sample(names, sample_count), key=image_sort_key)
    return [image_dir / name for name in selected_names]


def resolve_image_path(image: Path, image_dir: Path) -> Path:
    if image.exists():
        return image
    if not image.is_absolute() and image.parent == Path("."):
        candidate = image_dir / image.name
        if candidate.exists():
            return candidate
    return image


def summarize_timings(traces: list[dict[str, Any]], model_load_timings: dict[str, float]) -> dict[str, Any]:
    keys = ("siglip2_image_and_topk_sec", "qwen_sec", "fusion_and_selection_sec", "total_image_sec")
    totals = {key: float(sum(float(trace["timings_sec"].get(key, 0.0)) for trace in traces)) for key in keys}
    rows = len(traces)
    averages = {key.replace("_sec", "_avg_sec"): (value / rows if rows else 0.0) for key, value in totals.items()}
    load_total = float(model_load_timings.get("load_models_and_text_sec", 0.0))
    grand_total = float(load_total + totals["total_image_sec"])
    return {
        "model_load_sec": model_load_timings,
        "per_image_totals_sec": totals,
        "per_image_averages_sec": averages,
        "total_eval_wall_accounted_sec": grand_total,
        "total_siglip2_sec": float(
            model_load_timings.get("load_siglip2_sec", 0.0)
            + model_load_timings.get("text_embeddings_sec", 0.0)
            + totals["siglip2_image_and_topk_sec"]
        ),
        "total_qwen_sec": float(model_load_timings.get("load_qwen_sec", 0.0) + totals["qwen_sec"]),
        "total_fusion_and_selection_sec": totals["fusion_and_selection_sec"],
    }


def summarize_skip_vlm(traces: list[dict[str, Any]]) -> dict[str, Any]:
    rows = len(traces)
    skipped = sum(1 for trace in traces if bool((trace.get("skip_vlm") or {}).get("skipped")))
    rel_gaps = [float((trace.get("skip_vlm") or {}).get("visual_rel_gap", 0.0)) for trace in traces]
    return {
        "enabled": any(bool((trace.get("skip_vlm") or {}).get("enabled")) for trace in traces),
        "skipped_rows": skipped,
        "skip_rate": (skipped / rows) if rows else 0.0,
        "visual_rel_gap_avg": (sum(rel_gaps) / rows) if rows else 0.0,
    }


def build_moe_expert_usage_summary(runtime: DemoRuntime, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.moe_expert_usage_json is None:
        return None
    if runtime.qwen is None or runtime.qwen.moe_tracker is None:
        return {
            "schema_version": "moe_expert_usage_v1",
            "model_id": args.qwen_model,
            "enabled": False,
            "warning": "MoE expert usage was requested, but no VLM was loaded.",
        }
    return runtime.qwen.moe_tracker.summary(args.qwen_model)


def write_moe_expert_usage_summary(summary: dict[str, Any] | None, path: Path | None) -> None:
    if summary is None or path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def fmt_sec(value: float | int) -> str:
    return f"{float(value):.3f}s"


def print_single_timing(report: dict[str, Any]) -> None:
    timings = report["timings_sec"]
    summary = report["timing_summary_sec"]
    text_cache = report.get("text_cache", {})
    cache_state = "hit" if text_cache.get("hit") else "miss"
    print("Timing:")
    print(
        "- model load: "
        f"{fmt_sec(timings.get('load_models_and_text_sec', 0.0))} "
        f"(SigLIP2 {fmt_sec(timings.get('load_siglip2_sec', 0.0))}, "
        f"Qwen {fmt_sec(timings.get('load_qwen_sec', 0.0))}, "
        f"text {fmt_sec(timings.get('text_embeddings_sec', 0.0))} cache {cache_state})"
    )
    print(
        "- inference: "
        f"{fmt_sec(timings.get('total_image_sec', 0.0))} "
        f"(SigLIP2 image/top-k {fmt_sec(timings.get('siglip2_image_and_topk_sec', 0.0))}, "
        f"Qwen generate {fmt_sec(timings.get('qwen_sec', 0.0))}, "
        f"fusion/select {fmt_sec(timings.get('fusion_and_selection_sec', 0.0))})"
    )
    print(f"- total accounted: {fmt_sec(summary.get('total_eval_wall_accounted_sec', 0.0))}")


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "truth",
                "prediction",
                "f1",
                "siglip2_image_and_topk_sec",
                "qwen_sec",
                "fusion_and_selection_sec",
                "total_image_sec",
                "raw_qwen",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_eval_metrics(metrics: dict[str, Any]) -> None:
    print("Final metrics:")
    print(
        "- f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} rowavg_f1={rowavg_f1:.4f}".format(
            f1=float(metrics["f1"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            rowavg_f1=float(metrics["rowavg_f1"]),
        )
    )
    print(
        "- rows={rows} exact={exact}/{rows} mean_pred={mean_pred:.2f} mean_truth={mean_truth:.2f}".format(
            rows=int(metrics["rows"]),
            exact=int(metrics["exact_match_rows"]),
            mean_pred=float(metrics["mean_predicted_labels"]),
            mean_truth=float(metrics["mean_ground_truth_labels"]),
        )
    )


def zero_model_load_timings() -> dict[str, float]:
    return {
        "load_siglip2_sec": 0.0,
        "text_embeddings_sec": 0.0,
        "load_qwen_sec": 0.0,
        "load_models_and_text_sec": 0.0,
    }


def load_runtime(args: argparse.Namespace) -> DemoRuntime:
    cleaned_rows = read_cleaned_rows(args.cleaned_json)
    labels = read_labels(args.captions, cleaned_rows)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    gt_row_map = read_ground_truth_row_map(args.image_ground_truth_map, cleaned_rows)

    t0 = time.perf_counter()
    embedder = OpenCLIPSigLIP2Embedder(args.siglip_model, args.siglip_pretrained, args.device, args.torch_dtype)
    load_siglip2_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    text_embeddings, text_cache = load_or_build_text_embeddings(
        embedder,
        labels,
        args.siglip_model,
        args.siglip_pretrained,
        args.text_cache,
        use_cache=not args.no_text_cache,
    )
    text_embeddings_sec = time.perf_counter() - t0

    qwen = None
    load_qwen_sec = 0.0
    if not args.mock_qwen_json:
        t0 = time.perf_counter()
        qwen = VisionLanguageScorer(
            args.qwen_model,
            args.vlm_family,
            args.device,
            args.torch_dtype,
            args.qwen_min_pixels,
            args.qwen_max_pixels,
            args.qwen_max_new_tokens,
            track_moe_experts=args.moe_expert_usage_json is not None,
            moe_expert_top_k=args.moe_expert_top_k,
        )
        load_qwen_sec = time.perf_counter() - t0
    model_load_timings = {
        "load_siglip2_sec": load_siglip2_sec,
        "text_embeddings_sec": text_embeddings_sec,
        "load_qwen_sec": load_qwen_sec,
        "load_models_and_text_sec": load_siglip2_sec + text_embeddings_sec + load_qwen_sec,
    }
    return DemoRuntime(
        args=args,
        cleaned_rows=cleaned_rows,
        labels=labels,
        label_to_id=label_to_id,
        gt_row_map=gt_row_map,
        text_embeddings=text_embeddings,
        text_cache=text_cache,
        embedder=embedder,
        qwen=qwen,
        model_load_timings=model_load_timings,
    )


def build_single_report(
    runtime: DemoRuntime,
    image_path: Path,
    args: argparse.Namespace,
    model_load_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    load_timings = model_load_timings if model_load_timings is not None else runtime.model_load_timings
    report = run_one(image_path, runtime.labels, runtime.text_embeddings, runtime.embedder, runtime.qwen, args)
    report["label_count"] = len(runtime.labels)
    report["text_cache"] = runtime.text_cache
    report["timings_sec"].update(load_timings)
    report["timing_summary_sec"] = summarize_timings([report], load_timings)
    return report


def write_single_report(report: dict[str, Any], image_path: Path, args: argparse.Namespace) -> Path:
    out_path = args.output_json or args.output_dir / f"{image_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def print_single_report(report: dict[str, Any], out_path: Path) -> None:
    print("Selected ingredients:")
    for label in report["selected_labels"]:
        print(f"- {label}")
    print_single_timing(report)
    runtime_mode = report.get("runtime_mode")
    if runtime_mode:
        print(f"Runtime mode: {runtime_mode}")
    print(f"JSON trace: {out_path}")


def apply_request_overrides(base_args: argparse.Namespace, values: dict[str, Any]) -> argparse.Namespace:
    request_args = argparse.Namespace(**vars(base_args))
    for key in SERVER_REQUEST_ARG_FIELDS:
        if key in values:
            setattr(request_args, key, values[key])
    return request_args


def server_base_url(args: argparse.Namespace) -> str:
    return f"http://{args.server_host}:{args.server_port}"


def post_json(url: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            message = payload.get("message") or payload.get("error") or raw
        except Exception:
            message = raw
        raise RuntimeError(f"Server returned HTTP {exc.code}: {message}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"Server returned {type(parsed).__name__}, expected object")
    return parsed


def get_json(url: str, timeout: float | None = None) -> dict[str, Any]:
    with urllib_request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"Server returned {type(parsed).__name__}, expected object")
    return parsed


def server_request_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in SERVER_REQUEST_ARG_FIELDS}


def infer_via_server(args: argparse.Namespace, image_path: Path, output_json: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image": str(resolve_image_path(image_path, args.image_dir).resolve()),
        "args": server_request_args(args),
    }
    if output_json is not None:
        payload["output_json"] = str(output_json)
    return post_json(f"{server_base_url(args)}/infer", payload, timeout=None)


def try_server_health(args: argparse.Namespace) -> bool:
    try:
        get_json(f"{server_base_url(args)}/health", timeout=1.0)
        return True
    except Exception:
        return False


def shutdown_server(args: argparse.Namespace) -> None:
    try:
        response = post_json(f"{server_base_url(args)}/shutdown", {}, timeout=5.0)
    except Exception as exc:
        raise SystemExit(f"Could not shut down demo server at {server_base_url(args)}: {exc}") from exc
    print(response.get("message", "Shutdown requested."))


def run_single_via_server(args: argparse.Namespace) -> bool:
    if args.image is None:
        return False
    image_path = resolve_image_path(args.image, args.image_dir)
    out_path = args.output_json or args.output_dir / f"{image_path.stem}.json"
    try:
        report = infer_via_server(args, image_path, out_path)
    except Exception as exc:
        if args.server_required:
            raise SystemExit(f"Could not connect to demo server at {server_base_url(args)}: {exc}") from exc
        print(f"Demo server failed at {server_base_url(args)}: {exc}; falling back to local model load.")
        return False
    print_single_report(report, out_path)
    return True


def run_eval_via_server(args: argparse.Namespace) -> bool:
    if args.eval_samples <= 0:
        return False
    try:
        if not try_server_health(args):
            raise ConnectionError(f"No ready server at {server_base_url(args)}")
    except Exception as exc:
        if args.server_required:
            raise SystemExit(f"Could not connect to demo server at {server_base_url(args)}: {exc}") from exc
        print(f"Demo server failed at {server_base_url(args)}: {exc}; falling back to local model load.")
        return False

    cleaned_rows = read_cleaned_rows(args.cleaned_json)
    labels = read_labels(args.captions, cleaned_rows)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    gt_row_map = read_ground_truth_row_map(args.image_ground_truth_map, cleaned_rows)
    image_paths = list_eval_images(args.image_dir, args.images_list, args.eval_samples, args.seed, args.eval_first)
    predictions: list[set[int]] = []
    truths: list[set[int]] = []
    csv_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    measurement_enabled = bool(args.measure or args.measure_power)
    power_monitor = PowerMonitor(enabled=args.measure_power, interval_ms=args.power_sample_interval_ms)
    measured_wall_sec = 0.0
    power_summary: dict[str, Any] = {}
    if measurement_enabled:
        power_monitor.start()
    measurement_t0 = time.perf_counter()
    try:
        for idx, image_path in enumerate(image_paths, start=1):
            report = infer_via_server(args, image_path)
            pred = set(int(x) for x in report["selected_label_ids"])
            truth = set(ground_truth_ids_for_image(image_path.name, cleaned_rows, label_to_id, gt_row_map))
            predictions.append(pred)
            truths.append(truth)
            denom = 2 * len(pred & truth) + len(pred - truth) + len(truth - pred)
            row_f1 = (2 * len(pred & truth) / denom) if denom else 1.0
            csv_rows.append(
                {
                    "image": image_path.name,
                    "truth": "|".join(labels[i] for i in sorted(truth)),
                    "prediction": "|".join(labels[i] for i in sorted(pred)),
                    "f1": f"{row_f1:.6f}",
                    "siglip2_image_and_topk_sec": f"{report['timings_sec']['siglip2_image_and_topk_sec']:.6f}",
                    "qwen_sec": f"{report['timings_sec']['qwen_sec']:.6f}",
                    "fusion_and_selection_sec": f"{report['timings_sec']['fusion_and_selection_sec']:.6f}",
                    "total_image_sec": f"{report['timings_sec']['total_image_sec']:.6f}",
                    "raw_qwen": report["qwen"]["raw_text"],
                }
            )
            traces.append(report)
            print(
                f"[{idx}/{len(image_paths)}] {image_path.name} "
                f"f1={row_f1:.4f} time={report['timings_sec']['total_image_sec']:.3f}s "
                f"qwen={report['timings_sec']['qwen_sec']:.3f}s pred={report['selected_labels']}"
            )
    finally:
        measured_wall_sec = time.perf_counter() - measurement_t0
        if measurement_enabled:
            power_summary = power_monitor.stop(wall_sec=measured_wall_sec)

    metrics = metrics_from_sets(predictions, truths)
    summary = {
        "schema_version": "orin_cuda_siglip2_qwen3vl_eval_v1",
        "runtime_mode": "server_client",
        "server_url": server_base_url(args),
        "label_count": len(labels),
        "image_count": len(image_paths),
        "eval_selection": "first" if args.eval_first else "seeded_random",
        "seed": args.seed,
        "metrics": metrics,
        "traces": traces,
        "timing_summary_sec": summarize_timings(traces, zero_model_load_timings()),
        "skip_vlm_summary": summarize_skip_vlm(traces),
    }
    if measurement_enabled:
        latencies = [float(trace["timings_sec"].get("total_image_sec", 0.0)) for trace in traces]
        summary["benchmark"] = build_benchmark_summary(
            task_name="task1_mm_food100k",
            query_count=len(image_paths),
            latencies_sec=latencies,
            measured_wall_sec=measured_wall_sec,
            task_metric_name="f1",
            task_metric_value=float(metrics["f1"]),
            power_summary=power_summary,
            w_config=args.w_config,
            nvpmodel=query_nvpmodel(),
            extra_task_metrics={
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "rowavg_f1": metrics["rowavg_f1"],
            },
        )
    eval_name = f"eval_first_{len(image_paths)}_samples" if args.eval_first else f"eval_{len(image_paths)}_samples"
    out_path = args.output_json or args.output_dir / f"{eval_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.predictions_csv or args.output_dir / f"{eval_name}_predictions.csv"
    write_predictions_csv(csv_path, csv_rows)
    print_eval_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Timing summary:")
    print(json.dumps(summary["timing_summary_sec"], indent=2, sort_keys=True))
    if summary["skip_vlm_summary"]["enabled"]:
        print("Skip VLM summary:")
        print(json.dumps(summary["skip_vlm_summary"], indent=2, sort_keys=True))
    if "benchmark" in summary:
        print_benchmark_summary(summary["benchmark"])
    print(f"Evaluation JSON: {out_path}")
    print(f"Predictions CSV: {csv_path}")
    return True


def serve_runtime(runtime: DemoRuntime, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: Any) -> None:
            print(f"[server] {self.address_string()} - {format % values}")

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def read_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("JSON payload must be an object")
            return payload

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "not_found"})
                return
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model_load_sec": runtime.model_load_timings,
                    "settings": {key: getattr(runtime.args, key) for key in SERVER_REQUEST_ARG_FIELDS},
                },
            )

        def do_POST(self) -> None:
            try:
                if self.path == "/shutdown":
                    self.send_json(200, {"message": "Shutdown requested."})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                if self.path != "/infer":
                    self.send_json(404, {"error": "not_found"})
                    return
                payload = self.read_payload()
                raw_image = payload.get("image")
                if not raw_image:
                    self.send_json(400, {"error": "missing image"})
                    return
                request_args = apply_request_overrides(runtime.args, dict(payload.get("args") or {}))
                report = build_single_report(runtime, Path(str(raw_image)), request_args, zero_model_load_timings())
                report["runtime_mode"] = "server"
                report["server_model_load_sec"] = runtime.model_load_timings
                raw_output_json = payload.get("output_json")
                if raw_output_json:
                    output_path = Path(str(raw_output_json))
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                self.send_json(200, report)
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, end="")
                self.send_json(500, {"error": type(exc).__name__, "message": str(exc), "traceback": tb})

    httpd = HTTPServer((args.server_host, args.server_port), DemoHandler)
    print(f"Demo server ready at http://{args.server_host}:{args.server_port}")
    print("Models are loaded and will stay in memory until you stop this process with Ctrl+C or --shutdown-server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        httpd.server_close()


def main() -> None:
    args = parse_args()
    if args.shutdown_server:
        shutdown_server(args)
        return

    should_try_server = args.use_server or (
        not args.serve
        and (args.image is not None or args.eval_samples > 0)
        and try_server_health(args)
    )
    if should_try_server:
        if args.image is not None and run_single_via_server(args):
            return
        if args.eval_samples > 0 and run_eval_via_server(args):
            return

    if args.image is None and args.eval_samples <= 0 and not args.serve and not args.no_free_memory:
        raise SystemExit("Provide --image, --eval-samples N, --serve, or --noFreeMemory.")

    runtime = load_runtime(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is None and args.eval_samples <= 0:
        serve_runtime(runtime, args)
        return

    if args.image is not None:
        image_path = resolve_image_path(args.image, args.image_dir)
        report = build_single_report(runtime, image_path, args)
        moe_usage = build_moe_expert_usage_summary(runtime, args)
        if moe_usage is not None:
            report["moe_expert_usage"] = moe_usage
            write_moe_expert_usage_summary(moe_usage, args.moe_expert_usage_json)
        out_path = write_single_report(report, image_path, args)
        print_single_report(report, out_path)
        if args.moe_expert_usage_json is not None:
            print(f"MoE expert usage JSON: {args.moe_expert_usage_json}")
        if args.serve or args.no_free_memory:
            serve_runtime(runtime, args)
        return

    image_paths = list_eval_images(args.image_dir, args.images_list, args.eval_samples, args.seed, args.eval_first)
    predictions: list[set[int]] = []
    truths: list[set[int]] = []
    csv_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    measurement_enabled = bool(args.measure or args.measure_power)
    power_monitor = PowerMonitor(enabled=args.measure_power, interval_ms=args.power_sample_interval_ms)
    measured_wall_sec = 0.0
    power_summary: dict[str, Any] = {}
    if measurement_enabled:
        power_monitor.start()
    measurement_t0 = time.perf_counter()
    try:
        for idx, image_path in enumerate(image_paths, start=1):
            report = build_single_report(runtime, image_path, args, zero_model_load_timings())
            pred = set(int(x) for x in report["selected_label_ids"])
            truth = set(ground_truth_ids_for_image(image_path.name, runtime.cleaned_rows, runtime.label_to_id, runtime.gt_row_map))
            predictions.append(pred)
            truths.append(truth)
            denom = 2 * len(pred & truth) + len(pred - truth) + len(truth - pred)
            row_f1 = (2 * len(pred & truth) / denom) if denom else 1.0
            csv_rows.append(
                {
                    "image": image_path.name,
                    "truth": "|".join(runtime.labels[i] for i in sorted(truth)),
                    "prediction": "|".join(runtime.labels[i] for i in sorted(pred)),
                    "f1": f"{row_f1:.6f}",
                    "siglip2_image_and_topk_sec": f"{report['timings_sec']['siglip2_image_and_topk_sec']:.6f}",
                    "qwen_sec": f"{report['timings_sec']['qwen_sec']:.6f}",
                    "fusion_and_selection_sec": f"{report['timings_sec']['fusion_and_selection_sec']:.6f}",
                    "total_image_sec": f"{report['timings_sec']['total_image_sec']:.6f}",
                    "raw_qwen": report["qwen"]["raw_text"],
                }
            )
            traces.append(report)
            print(
                f"[{idx}/{len(image_paths)}] {image_path.name} "
                f"f1={row_f1:.4f} time={report['timings_sec']['total_image_sec']:.3f}s "
                f"qwen={report['timings_sec']['qwen_sec']:.3f}s pred={report['selected_labels']}"
            )
    finally:
        measured_wall_sec = time.perf_counter() - measurement_t0
        if measurement_enabled:
            power_summary = power_monitor.stop(wall_sec=measured_wall_sec)

    metrics = metrics_from_sets(predictions, truths)
    summary = {
        "schema_version": "orin_cuda_siglip2_qwen3vl_eval_v1",
        "label_count": len(runtime.labels),
        "image_count": len(image_paths),
        "eval_selection": "first" if args.eval_first else "seeded_random",
        "seed": args.seed,
        "models": {
            "siglip_model": args.siglip_model,
            "siglip_pretrained": args.siglip_pretrained,
            "qwen_model": args.qwen_model,
            "vlm_model": args.qwen_model,
            "vlm_family": resolve_vlm_family(args.qwen_model, args.vlm_family),
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "image_ground_truth_map": {
            "path": str(args.image_ground_truth_map),
            "loaded_rows": len(runtime.gt_row_map),
        },
        "text_cache": runtime.text_cache,
        "metrics": metrics,
        "traces": traces,
        "timing_summary_sec": summarize_timings(traces, runtime.model_load_timings),
        "skip_vlm_summary": summarize_skip_vlm(traces),
    }
    moe_usage = build_moe_expert_usage_summary(runtime, args)
    if moe_usage is not None:
        summary["moe_expert_usage"] = moe_usage
        write_moe_expert_usage_summary(moe_usage, args.moe_expert_usage_json)
    if measurement_enabled:
        latencies = [float(trace["timings_sec"].get("total_image_sec", 0.0)) for trace in traces]
        summary["benchmark"] = build_benchmark_summary(
            task_name="task1_mm_food100k",
            query_count=len(image_paths),
            latencies_sec=latencies,
            measured_wall_sec=measured_wall_sec,
            task_metric_name="f1",
            task_metric_value=float(metrics["f1"]),
            power_summary=power_summary,
            w_config=args.w_config,
            nvpmodel=query_nvpmodel(),
            extra_task_metrics={
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "rowavg_f1": metrics["rowavg_f1"],
            },
        )
    eval_name = f"eval_first_{len(image_paths)}_samples" if args.eval_first else f"eval_{len(image_paths)}_samples"
    out_path = args.output_json or args.output_dir / f"{eval_name}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.predictions_csv or args.output_dir / f"{eval_name}_predictions.csv"
    write_predictions_csv(csv_path, csv_rows)
    print_eval_metrics(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Timing summary:")
    print(json.dumps(summary["timing_summary_sec"], indent=2, sort_keys=True))
    if summary["skip_vlm_summary"]["enabled"]:
        print("Skip VLM summary:")
        print(json.dumps(summary["skip_vlm_summary"], indent=2, sort_keys=True))
    if "benchmark" in summary:
        print_benchmark_summary(summary["benchmark"])
    print(f"Evaluation JSON: {out_path}")
    print(f"Predictions CSV: {csv_path}")
    if args.moe_expert_usage_json is not None:
        print(f"MoE expert usage JSON: {args.moe_expert_usage_json}")
    if args.serve or args.no_free_memory:
        serve_runtime(runtime, args)


if __name__ == "__main__":
    main()
