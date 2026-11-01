# Commands

Every `uv` command takes exactly one CUDA profile: `--extra cu126` (baseline) or `--extra cu128` (Blackwell,
e.g. RTX 5090). The examples use cu126; swap in cu128 consistently. The host is shared: pick `<gpu>` from
`nvidia-smi` and `<port>` from the free ports (8000 is taken; the container binds 18000).

## Setup

```bash
uv sync --extra cu126                                   # Python 3.12; torch 2.14.1+cu126 (cu128: 2.11.0)
uv run --extra cu126 scripts/stage_models.py            # pinned models -> ./models (--verify: offline hash check)
uv run --extra cu126 scripts/fetch_fixtures.py          # fixtures (--verify [--small-only]: hash check only)
uv run --extra cu126 pre-commit install                 # autoflake, isort, ruff-format, ruff (ruff.toml)
```

## Tests

```bash
uv run --extra cu126 pytest                             # CPU tiers; GPU/container tests skip without CUDA/docker
(cd tests/contract/js && npm ci && npm test)            # JS SDK contract (tsc --strict + node --test)
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 pytest tests/gpu
VIEGEM_GPU=<gpu> uv run --extra cu128 --offline pytest tests/container --container-cuda=cu128   # image gate
uv run --extra cu126 pre-commit run --all-files         # lint + format
```

The CPU tiers need `ffmpeg` on PATH. Image gate details (cu126 run, CPU diagnostics): docs/container-gate.vi.md.

## Local server and smoke

```bash
VIEGEM_API_KEYS=<key of >=16 chars> CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 \
    uvicorn --factory app.main:create_app --port <port> --workers 1 --no-access-log \
    --timeout-graceful-shutdown 30 --ws websockets-sansio --ws-max-size 2097152    # /ready after warm-up
OPENAI_API_KEY=<key> uv run --extra cu126 scripts/smoke_server.py \
    --base-url http://127.0.0.1:<port> --out docs/baseline/server.json
```

Listening page against a running gateway (writes `outputs/demo/index.html`; `outputs/` is gitignored):

```bash
OPENAI_API_KEY=<key> uv run --extra cu126 scripts/demo_page.py --base-url http://127.0.0.1:<port> \
    [--add <audio file or dir>] [--only <case ids or categories>] [--out outputs/<name>]
```

If `uv run ... uvicorn` fails with `Failed to spawn`, the venv was created under another path (stale shebangs):
use `uv run --extra cu126 python -m uvicorn ...` or recreate `.venv`.

One process per GPU, always one worker. Start the server from a shell with no `VIEGEM_GPU`, `VIEGEM_BUILD_CUDA` or
`VIEGEM_TEST_*` exported: unknown `VIEGEM_*` variables abort startup.

## Baselines and probes

```bash
CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$(mktemp -d) uv run --extra cu126 --offline scripts/smoke_inference.py \
    --no-network --out docs/baseline/offline.json
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 scripts/probe_alignment.py --out docs/baseline/alignment_probe.json
```

Diagnostic precision and silence measurements (do not change serving defaults):

```bash
# Serving precision is fp32 (the default). Each precision writes its own report; bf16/fp32 exit 1 on the
# recorded QUAN fixture-WER limit (accepted exception, #4); fp16 exits 1 on overflow.
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 --offline scripts/smoke_inference.py \
    --precision fp32 --no-network --out docs/baseline/precision-fp32.json   # also bf16, fp16
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 --offline scripts/probe_silence.py \
    --out docs/baseline/silence_probe.json

# Public pinned dev set (~215 MB); only this step downloads data, into gitignored outputs/.
uv run --extra cu126 hf download google/fleurs data/vi_vn/dev.tsv data/vi_vn/audio/dev.tar.gz \
    --repo-type dataset --revision 70bb2e84b976b7e960aa89f1c648e09c59f894dd --local-dir outputs/fleurs-dev
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 --offline scripts/eval_fleurs.py \
    --data-dir outputs/fleurs-dev/data/vi_vn --precision bf16 --out docs/baseline/fleurs-dev-bf16.json
# Repeat with --precision fp32 and a separate output file.
```

The FLEURS/silence scripts block network, verify staged model hashes and save metadata only.
The local historical-QUAN harness and paired-bootstrap reproduction are retained in
`outputs/precision/`; their scope/results are documented in docs/baseline.vi.md §8.
This is exploratory evidence, not a passed release gate.

Benchmark invocation (R4): docs/roadmap.md §1.
