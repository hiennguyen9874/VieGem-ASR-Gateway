"""Load QUAN-Transcribe from staged local directories only.

`QuanTranscribe.from_pretrained()` (quan-transcribe@88ddcd2) loads the base encoder, feature extractor and tokenizer
from `config.json:encoder_id` / `units.json:model_id` (a Hub id resolved at the `main` branch), so its `revision`
argument does not pin them. This loader mirrors that function step by step but takes the staged base directory
explicitly, so nothing is resolved against the Hub. Parity with the reference loader: tests/gpu/test_loader_parity.py.
"""

import dataclasses
import json
from pathlib import Path

import torch
from quan_transcribe import QuanTranscribe
from quan_transcribe.model import ModelConfig, QuanCTC, load_encoder
from safetensors.torch import load_file
from transformers import AutoTokenizer

QUAN_DIR = "quan-transcribe-1.0"
BASE_DIR = "embeddinggemma-2"
BASE_ID = "google/embeddinggemma-2"


def load_quan(models_dir: str | Path, device: str | torch.device) -> QuanTranscribe:
    """models_dir: staging root of scripts/stage_models.py (holds quan-transcribe-1.0/ and embeddinggemma-2/)."""
    quan_dir, base_dir = Path(models_dir) / QUAN_DIR, Path(models_dir) / BASE_DIR
    cfg = ModelConfig.from_dict(json.loads((quan_dir / "config.json").read_text()))
    units = json.loads((quan_dir / "units.json").read_text())
    # The staged base must be the asset the checkpoint was trained on.
    for key, value in (("config.json:encoder_id", cfg.encoder_id), ("units.json:model_id", units.get("model_id"))):
        if value != BASE_ID:
            raise RuntimeError(f"unexpected base model {key}={value!r}, staged base is {BASE_ID}")
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model not staged: {base_dir} (run scripts/stage_models.py)")

    # LogMel and load_encoder() read cfg.encoder_id; point it at the local copy.
    cfg = dataclasses.replace(cfg, encoder_id=str(base_dir))
    device = torch.device(device)
    model = QuanCTC(cfg, load_encoder(cfg.encoder_id))
    state = load_file(quan_dir / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    return QuanTranscribe(model, tokenizer, units["gemma_ids"], device)
