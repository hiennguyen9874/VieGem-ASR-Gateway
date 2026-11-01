"""Load the pinned VieGem checkpoint and base encoder from staged local files only."""

import dataclasses
from pathlib import Path

import torch
from transformers import AutoTokenizer

from app.viegem.model import CTCModel, ModelConfig
from app.viegem.text import Units

BASE_ID = "google/embeddinggemma-2"
BASE_DIR = "embeddinggemma-2"
VIEGEM_DIR = "viegem-asr"


def load_viegem(models_dir: str | Path, device: str | torch.device):
    root = Path(models_dir)
    base_dir = root / BASE_DIR
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model not staged: {base_dir} (run scripts/stage_models.py)")
    checkpoint = torch.load(
        root / VIEGEM_DIR / "viegem_asr.pt", map_location="cpu", weights_only=True
    )
    encoder_id = checkpoint["model_config"].get("encoder_id")
    if encoder_id != BASE_ID:
        raise RuntimeError(
            f"unexpected base model encoder_id={encoder_id!r}, staged base is {BASE_ID}"
        )
    cfg = ModelConfig.from_dict(checkpoint["model_config"])
    cfg = dataclasses.replace(cfg, encoder_id=str(base_dir.resolve()))
    units = Units(checkpoint["gemma_ids"])
    model = CTCModel(cfg, units.gemma_ids)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in missing if not key.startswith("encoder.")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: missing={missing} unexpected={unexpected}"
        )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir.resolve()), local_files_only=True)
    return model, tokenizer, units
