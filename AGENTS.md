# Deploy QUAN-Transcribe-1.0

- Design: docs/deployment-design.vi.md
- Plan: docs/implementation-plan.vi.md (M3 REST/SSE done; container gate not yet run: no Docker on this host)
- M3 design + deviations: docs/m3-design.vi.md
- Baseline (M0 results, pins, measurements): docs/baseline.vi.md
- API contract (pinned OpenAI SDK profile, payloads, error codes): docs/api-contract.vi.md

## Project structure

```text
app/quan_loader.py         # load QUAN from staged local dirs only (no Hub resolution); reuses reference classes
app/config.py              # typed QUAN_* env config, validated at startup (no default API key)
app/engine.py              # QuanEngine adapter + EngineWorker: the single thread that runs the model
app/scheduler.py           # bounded fair queue (lanes = requests/sessions, jobs = chunks) feeding the worker
app/audio.py               # multipart ingress (body limit before parser, bounded spool) + sandboxed FFmpeg decode
                           #   + decoded-audio RAM budget (`hold`)
app/chunking.py            # quiet-point splitter: contiguous spans, <=100 ms audio -> no chunk (empty transcript)
app/main.py                # FastAPI app factory: lifespan, /health, /ready, /metrics, auth, request ids
app/transcription.py       # /v1/models, POST /v1/audio/transcriptions (raw ASGI: json/text/SSE, disconnect watch,
                           #   send timeout), parse_params, Joiner (the only text joiner), run_chunks
app/errors.py, schemas.py  # OpenAI error envelope + status/code mapping
app/observability.py       # JSON logs + Prometheus-style metrics (metadata only)
assets/manifest.json       # single source of model pins: repo id, commit, size + SHA256 per file
scripts/stage_models.py    # download pinned assets into ./models (or $QUAN_MODELS_DIR), verify SHA256
scripts/fetch_fixtures.py  # build fixtures from FLEURS vi_vn (CC BY 4.0) at a pinned revision, verify SHA256
scripts/smoke_inference.py # inference baseline: full/streaming, timings, RAM/VRAM, short-clip probe -> JSON
scripts/smoke_server.py    # stdlib smoke of a running gateway (local/container) vs the offline baseline -> JSON
Dockerfile, .dockerignore, compose.yaml  # R1 image (pinned digests, non-root, models mounted ro, no secrets)
docs/baseline/*.json       # saved smoke reports (online / offline / server)
tests/
  unit/                    # CPU: manifests and pins, config, app over real Uvicorn with a fake engine;
                           #      transcription params/joiner and end-to-end API (FFmpeg, disconnect, SSE, drain)
  scheduler/               # CPU: scheduler/worker gate with a fake engine (serialization, bounds, fairness, cancel)
  audio/                   # CPU + ffmpeg: formats, hostile inputs, decode bounds, multipart limits, temp cleanup
  fake_engine.py           # engine stand-in shared by CPU tests (`text_fn` for real decoded audio)
  live_server.py           # real Uvicorn around create_app + float-WAV audio helpers, shared by E2E/contract tests
  contract/                # CPU: examples/ parsed by pinned OpenAI Python SDK; js/ for the JS SDK;
                           #      test_server_sdk.py: real server vs examples, Python SDK + js/server.test.ts
  gpu/                     # GPU: local loader == reference loader (exact logits, needs network); real engine;
                           #      decode -> chunk -> engine pipeline; test_server.py: real app over HTTP
  container/               # Docker: image gate (skips without docker; runtime part needs QUAN_GPU)
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
uv run pytest                                    # unit + scheduler + contract; GPU tests skip without CUDA/assets
CUDA_VISIBLE_DEVICES=<free gpu> uv run pytest tests/gpu
QUAN_API_KEYS=<key> CUDA_VISIBLE_DEVICES=<free gpu> uv run uvicorn --factory app.main:create_app \
    --workers 1 --no-access-log --timeout-graceful-shutdown 30   # one process per GPU; /ready after warm-up
(cd tests/contract/js && npm ci && npm test)     # JS SDK contract (tsc --strict + node --test); server tests via pytest
OPENAI_API_KEY=<key> uv run scripts/smoke_server.py --base-url http://127.0.0.1:<port> --out docs/baseline/server.json
QUAN_GPU=<free gpu> uv run pytest tests/container   # needs docker + NVIDIA Container Toolkit
```

## Conventions

- The host GPUs are shared: check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` to a free GPU.
- Change a model pin only in `assets/manifest.json`, then run `stage_models.py --update-manifest`. Keep the QUAN code
  commit in `pyproject.toml` equal to the manifest (enforced by `tests/unit/test_manifests.py`).
- Load the model through `app/quan_loader.load_quan`, not `QuanTranscribe.from_pretrained` (that resolves the base
  model at Hub `main`). Do not patch code under `references/`.
- All inference on one model instance must be serialized (`QuanCTC._acts` is mutable hook state): only
  `app/engine.py` calls QUAN, and only via `Scheduler` lanes; endpoints never touch the model or the worker.
- Never forward waveforms shorter than 161 samples at 16 kHz (the frontend crashes); see docs/baseline.vi.md §7.
  Chunk audio only with `app/chunking.split`; decode client audio only with `AudioIngress.decode` (FFmpeg on PATH,
  needed by the CPU tests too). Never pass client filenames, paths or URLs to FFmpeg.
- The host is shared: port 8000 is taken by another service here, so the default is 18000; pick a free port for local servers.
- Transcription endpoints never call the model or the worker: lane per request, `run_chunks`, `Joiner`.
  `QUAN_GPU` (compose, container test) and `QUAN_TEST_*` (JS server test) are not server config: do not export
  them in a shell that starts the server; an unknown `QUAN_*` variable aborts startup.
- Contract changes: update `tests/contract/examples/` plus both SDK tests, then docs/api-contract.vi.md. Omit
  optional fields rather than sending `null` (the JS types reject `null` for several of them).
- Never commit or log HF_TOKEN, API keys, audio or transcripts.

## References

- audio.cpp: references/audio.cpp
- QUAN-Transcribe: references/QUAN-Transcribe
- QUAN-Transcribe-1.0: references/QUAN-Transcribe-1.0
- WhisperLiveKit: references/WhisperLiveKit
