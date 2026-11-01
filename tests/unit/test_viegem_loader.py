"""The future VieGem loader rejects incompatible checkpoints without accessing the Hub."""

from types import SimpleNamespace

import pytest
import torch

from app import viegem_loader


@pytest.fixture
def staged(tmp_path):
    (tmp_path / "embeddinggemma-2").mkdir()
    (tmp_path / "viegem-asr").mkdir()
    return tmp_path


@pytest.mark.parametrize("config", [{"encoder_id": "wrong"}, {}])
def test_rejects_wrong_encoder_before_model_load(staged, config):
    torch.save({"model_config": config}, staged / "viegem-asr/viegem_asr.pt")
    with pytest.raises(RuntimeError, match="unexpected base model"):
        viegem_loader.load_viegem(staged, "cpu")


@pytest.mark.parametrize("missing,unexpected", [(["head_bias"], []), ([], ["extra"])])
def test_rejects_incompatible_checkpoint(staged, monkeypatch, missing, unexpected):
    torch.save(
        {"model_config": {"encoder_id": viegem_loader.BASE_ID}, "gemma_ids": [], "model": {}},
        staged / "viegem-asr/viegem_asr.pt",
    )
    monkeypatch.setattr(
        viegem_loader,
        "CTCModel",
        lambda cfg, ids: SimpleNamespace(
            load_state_dict=lambda state, strict: (missing, unexpected)
        ),
    )
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        viegem_loader.load_viegem(staged, "cpu")


def test_rejects_checkpoint_requiring_pickle(staged):
    torch.save({"model_config": SimpleNamespace()}, staged / "viegem-asr/viegem_asr.pt")
    with pytest.raises(Exception, match="Weights only load failed"):
        viegem_loader.load_viegem(staged, "cpu")
