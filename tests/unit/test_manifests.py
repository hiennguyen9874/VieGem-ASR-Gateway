"""CPU lane: pins are exact commits and the committed fixtures match their manifest."""

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

from app.config import Config
from scripts.fetch_fixtures import MANIFEST as FIXTURES_MANIFEST
from scripts.fetch_fixtures import verify

ROOT = Path(__file__).resolve().parents[2]
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_asset_manifest_pins_commits_and_hashes():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    assert COMMIT.match(manifest["code"]["quan-transcribe"]["commit"])
    for entry in manifest["models"].values():
        assert COMMIT.match(entry["revision"])
        for meta in entry["files"].values():
            assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]) and meta["size"] > 0


def test_pyproject_pins_quan_code_to_manifest_commit():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    commit = json.loads((ROOT / "assets/manifest.json").read_text())["code"]["quan-transcribe"]["commit"]
    assert any(d.startswith("quan-transcribe @ git+") and d.endswith(f"@{commit}") for d in deps)


def test_cuda_profiles_are_exclusive_and_locked():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = {"cu126": "2.14.1", "cu128": "2.11.0"}
    assert project["tool"]["uv"]["conflicts"] == [[{"extra": "cu126"}, {"extra": "cu128"}]]
    indexes = {i["name"]: i for i in project["tool"]["uv"]["index"]}
    sources = {s["extra"]: s["index"] for s in project["tool"]["uv"]["sources"]["torch"]}
    locked = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    for cuda, version in expected.items():
        assert project["project"]["optional-dependencies"][cuda] == [f"torch=={version}"]
        index = indexes[sources[cuda]]
        assert index["explicit"] is True
        assert index["url"] == f"https://download.pytorch.org/whl/{cuda}"
        wheels = [p for p in locked if p["name"] == "torch" and p["version"] == f"{version}+{cuda}"]
        assert len(wheels) == 1
        assert wheels[0]["source"]["registry"] == index["url"]
        assert any("cp312-cp312" in w["url"] and "x86_64" in w["url"] for w in wheels[0]["wheels"])


def test_docker_cuda_profile_selection():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ARG TORCH_CUDA=cu126" in dockerfile
    assert 'case "$TORCH_CUDA" in cu126|cu128)' in dockerfile
    assert 'uv sync --frozen --no-dev --extra "$TORCH_CUDA"' in dockerfile
    compose = (ROOT / "compose.yaml").read_text()
    assert 'TORCH_CUDA: "${VIEGEM_BUILD_CUDA:-cu126}"' in compose


def test_compose_identity_and_server_environment():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    assert list(compose["services"]) == ["viegem"]
    service = compose["services"]["viegem"]
    assert service["image"].startswith("viegem-asr-gateway:")
    assert service["secrets"] == ["viegem_api_keys"]
    assert compose["secrets"]["viegem_api_keys"]["file"] == "./secrets/viegem_api_keys"
    env = service["environment"].copy()
    assert env.pop("VIEGEM_API_KEYS_FILE") == "/run/secrets/viegem_api_keys"
    env["VIEGEM_API_KEYS"] = "test-key-0123456789"
    cfg = Config.from_env(env)
    assert cfg.models_dir == Path("/models") and cfg.device == "cuda"


def test_staging_uses_viegem_models_dir_and_dest_override(tmp_path):
    env = {**os.environ, "VIEGEM_MODELS_DIR": str(tmp_path / "models")}
    env["QUAN_MODELS_DIR"] = str(tmp_path / "legacy")
    command = [sys.executable, "scripts/stage_models.py", "--verify", "--only", "quan-transcribe-1.0"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert str(tmp_path / "models" / "quan-transcribe-1.0") in result.stderr
    override = tmp_path / "other"
    result = subprocess.run([*command, "--dest", str(override)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert str(override / "quan-transcribe-1.0") in result.stderr


def test_small_fixtures_match_manifest():
    assert verify(json.loads(FIXTURES_MANIFEST.read_text()), include_large=False) == []
