"""CPU lane: pins are exact commits and the committed fixtures match their manifest."""

import json
import re
import tomllib
from pathlib import Path

from scripts.fetch_fixtures import MANIFEST as FIXTURES_MANIFEST, verify

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


def test_small_fixtures_match_manifest():
    assert verify(json.loads(FIXTURES_MANIFEST.read_text()), include_large=False) == []
