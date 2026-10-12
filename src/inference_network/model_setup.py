"""Provider-only preflight and explicit consent for the pinned model download."""

import importlib
import importlib.metadata
import json
import platform
from collections.abc import Callable
from pathlib import Path

from inference_network.config import MLX_MODEL_ID, MLX_MODEL_REVISION


def check_mlx_support() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The provider requires native Apple Silicon macOS")
    if int(platform.mac_ver()[0].split(".")[0]) < 26:
        raise RuntimeError("The pinned MLX runtime requires macOS 26 or newer")
    for package, version in (("mlx", "0.32.3"), ("mlx-lm", "0.32.0")):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            raise RuntimeError("Install the pinned provider dependencies first; see docs/setup.md")
    mx = importlib.import_module("mlx.core")
    if not mx.metal.is_available():
        raise RuntimeError("Apple Metal is unavailable on this provider")


def snapshot_complete(snapshot: Path) -> bool:
    if not all((snapshot / name).is_file() for name in ("config.json", "tokenizer_config.json")):
        return False
    if not (snapshot / "tokenizer.json").is_file():
        return False
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        try:
            weights = json.loads(index.read_text())["weight_map"]
            if not isinstance(weights, dict):
                return False
            shards = set(weights.values())
            return bool(shards) and all(
                isinstance(name, str) and Path(name).name == name and (snapshot / name).is_file()
                for name in shards
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return (snapshot / "model.safetensors").is_file()


def prepare_model(confirm: Callable[[str], str] = input) -> None:
    """The serving backend remains offline; only this consented setup may download."""
    check_mlx_support()
    hub = importlib.import_module("huggingface_hub")
    errors = importlib.import_module("huggingface_hub.errors")
    try:
        snapshot = hub.snapshot_download(
            MLX_MODEL_ID, revision=MLX_MODEL_REVISION, local_files_only=True, token=False
        )
    except errors.LocalEntryNotFoundError:
        snapshot = None
    if snapshot is not None and snapshot_complete(Path(snapshot)):
        print("Pinned model is cached; no download needed.")
        return
    answer = confirm(
        f"Pinned {MLX_MODEL_ID} weights are missing or incomplete (about 276 MiB). "
        "Download from Hugging Face now? [y/N]: "
    )
    if answer.strip().lower() not in {"y", "yes"}:
        raise RuntimeError("Model download declined; provider was not started")
    hub.snapshot_download(MLX_MODEL_ID, revision=MLX_MODEL_REVISION, token=False)
