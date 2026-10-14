# Deploy QUAN-Transcribe-1.0

- Design: docs/deployment-design.vi.md
- Plan: docs/implementation-plan.vi.md (M0 done)
- Baseline (M0 results, pins, measurements): docs/baseline.vi.md
- API contract (pinned OpenAI SDK profile, payloads, error codes): docs/api-contract.vi.md

## Project structure

```text
app/quan_loader.py         # load QUAN from staged local dirs only (no Hub resolution); reuses reference classes
assets/manifest.json       # single source of model pins: repo id, commit, size + SHA256 per file
scripts/stage_models.py    # download pinned assets into ./models (or $QUAN_MODELS_DIR), verify SHA256
scripts/fetch_fixtures.py  # build fixtures from FLEURS vi_vn (CC BY 4.0) at a pinned revision, verify SHA256
scripts/smoke_inference.py # inference baseline: full/streaming, timings, RAM/VRAM, short-clip probe -> JSON
docs/baseline/*.json       # saved smoke reports (online / offline)
tests/
  unit/                    # CPU: manifests and pins
  contract/                # CPU: examples/ parsed by pinned OpenAI Python SDK; js/ for the JS SDK
  gpu/                     # GPU + network: local loader == reference loader (exact logits)
  fixtures/                # small committed fixtures + manifest.json (hashes, references, license)
models/                    # staged assets (gitignored)
fixtures-large/            # large fixtures, e.g. vi_long.wav (gitignored)
references/                # read-only upstream sources; never import from or copy into images
```

## Commands

```bash
uv sync                                          # Python 3.12, torch 2.14.1+cu126, quan-transcribe@88ddcd2
uv run scripts/stage_models.py                   # stage pinned models (--verify: offline hash check only)
uv run scripts/fetch_fixtures.py                 # build fixtures (--verify [--small-only]: hash check only)
CUDA_VISIBLE_DEVICES=<free gpu> uv run scripts/smoke_inference.py --out docs/baseline/online.json
CUDA_VISIBLE_DEVICES=<free gpu> HF_HOME=$(mktemp -d) uv run --offline scripts/smoke_inference.py \
    --no-network --out docs/baseline/offline.json
uv run pytest                                    # unit + contract; GPU parity skips without CUDA/assets/network
(cd tests/contract/js && npm ci && npm test)     # JS SDK contract (tsc --strict + node --test)
```

## Conventions

- The host GPUs are shared: check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` to a free GPU.
- Change a model pin only in `assets/manifest.json`, then run `stage_models.py --update-manifest`. Keep the QUAN code
  commit in `pyproject.toml` equal to the manifest (enforced by `tests/unit/test_manifests.py`).
- Load the model through `app/quan_loader.load_quan`, not `QuanTranscribe.from_pretrained` (that resolves the base
  model at Hub `main`). Do not patch code under `references/`.
- All inference on one model instance must be serialized (`QuanCTC._acts` is mutable hook state).
- Never forward waveforms shorter than 161 samples at 16 kHz (the frontend crashes); see docs/baseline.vi.md §7.
- Contract changes: update `tests/contract/examples/` plus both SDK tests, then docs/api-contract.vi.md. Omit
  optional fields rather than sending `null` (the JS types reject `null` for several of them).
- Never commit or log HF_TOKEN, API keys, audio or transcripts.

## References

- audio.cpp: references/audio.cpp
- QUAN-Transcribe: references/QUAN-Transcribe
- QUAN-Transcribe-1.0: references/QUAN-Transcribe-1.0
- WhisperLiveKit: references/WhisperLiveKit
