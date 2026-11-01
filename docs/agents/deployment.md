# Pins, image and release evidence

- Model pins have one source of truth: `assets/manifest.json`. For a pin change, update the manifest and run `scripts/stage_models.py --update-manifest`; keep vendored code and the read-only VieGem reference at the manifest revision, and check GPU parity. See [baseline](../baseline.md) for measurements and licensing.
- Select exactly one CUDA extra; profile definitions are in `pyproject.toml`. Docker uses `TORCH_CUDA`, Compose uses `VIEGEM_BUILD_CUDA`, and container tests use `--container-cuda`.
- Keep the runtime image limited to `app/` and its venv; mount staged models read-only and supply keys via a Docker secret.
- After changing torch, CUDA, base image or FFmpeg pins, run unit and contract tests and the entire GPU image gate for each target profile without skips. Follow [testing.md](../testing.md#gate-docker-image) for prerequisites, commands and current evidence. CPU diagnostics and historical reports do not establish a GPU release gate; record missing prerequisites or failed/skipped checks rather than claiming a pass.
- Baseline reports contain metadata only; keep CPU-diagnostic container reports uncommitted.
