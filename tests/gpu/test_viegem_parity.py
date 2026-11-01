"""Exact-logits parity (fp32 and bf16 autocast) with the pinned read-only VieGem-ASR reference, using local assets."""

import dataclasses
import importlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("VIEGEM_MODELS_DIR", ROOT / "models"))
REFERENCE = ROOT / "references/VieGem-ASR"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        not (MODELS / "embeddinggemma-2").is_dir()
        or not (MODELS / "viegem-asr/viegem_asr.pt").is_file(),
        reason="assets not staged",
    ),
]


def test_viegem_local_loader_matches_reference(monkeypatch):
    from app.viegem_loader import load_viegem

    assert (REFERENCE / "asr/model.py").is_file(), "initialize the VieGem-ASR reference submodule"
    revision = json.loads((ROOT / "assets/manifest.json").read_text())["models"]["viegem-asr"][
        "revision"
    ]
    assert (
        subprocess.check_output(
            ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
        ).strip()
        == revision
    )
    monkeypatch.syspath_prepend(str(REFERENCE))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    ref = importlib.import_module("asr.model")
    ck = torch.load(MODELS / "viegem-asr/viegem_asr.pt", map_location="cpu", weights_only=True)
    cfg = dataclasses.replace(
        ref.ModelConfig.from_dict(ck["model_config"]),
        encoder_id=str((MODELS / "embeddinggemma-2").resolve()),
    )
    local, tokenizer, units = load_viegem(MODELS, "cuda")
    reference = ref.CTCModel(cfg, ck["gemma_ids"])
    missing, unexpected = reference.load_state_dict(ck["model"], strict=False)
    assert not [k for k in missing if not k.startswith("encoder.")] and not unexpected
    reference = reference.cuda().eval()
    assert units.gemma_ids == ck["gemma_ids"]
    assert local.state_dict().keys() == reference.state_dict().keys()
    for (name, param), (ref_name, ref_param) in zip(
        local.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name and param.requires_grad == ref_param.requires_grad
    for name in ("vi_a.wav", "vi_b.wav"):
        wav, sr = sf.read(ROOT / "tests/fixtures" / name, dtype="float32")
        assert sr == 16000
        x = torch.from_numpy(wav).unsqueeze(0).cuda()
        lengths = torch.tensor([x.shape[1]], device="cuda")
        # fp32 is the serving precision; bf16 autocast stays covered (its numerics depend on the grad flags).
        for streaming, bf16 in itertools.product((False, True), (False, True)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits, lens = local(x, lengths, streaming=streaming)
                expected, expected_lens = reference(x, lengths, streaming=streaming)
            assert torch.equal(lens, expected_lens) and torch.equal(logits, expected), (
                name,
                streaming,
                bf16,
            )
            assert units.decode(tokenizer, ref.greedy_decode(logits, lens)[0]) == units.decode(
                tokenizer, ref.greedy_decode(expected, expected_lens)[0]
            )
