# VieGem ASR Gateway

Docker-only Vietnamese ASR gateway with a scoped OpenAI REST/SSE and Realtime API. TLS, CORS and per-client rate limits belong to the operator's infrastructure.

## Always

- Pass exactly one `--extra cu126` or `--extra cu128` to every `uv sync` / `uv run`; otherwise uv can select unprofiled PyPI torch.
- The host is shared: choose a free GPU with `nvidia-smi` and a free server port (8000 is taken).
- A feature ships only after its gate passes with recorded evidence. Finish changes with code, tests and affected documentation in agreement; report any unrun checks.

## Task pointers

Read every applicable guide before editing:

- **Inference, scheduler, LM or vendored model:** [inference.md](docs/agents/inference.md).
- **Audio ingress, chunking or timestamps:** [audio.md](docs/agents/audio.md).
- **API, Realtime or configuration:** [api.md](docs/agents/api.md).
- **Model/dependency pins, image or release gate:** [deployment.md](docs/agents/deployment.md).
- **Setup, tests, local server, smoke or probes:** [commands.md](docs/agents/commands.md).
- **Find implementation or tests:** [code-map.md](docs/agents/code-map.md).
- **Domain terminology or architecture decisions:** [domain.md](docs/agents/domain.md).
- **GitHub issues or wayfinder:** [issue-tracker.md](docs/agents/issue-tracker.md); **triage labels:** [triage-labels.md](docs/agents/triage-labels.md).

## Project sources

- Overview and quick start: [README.md](README.md); deployment, usage, configuration, operations: [docs/deployment.md](docs/deployment.md), [docs/usage.md](docs/usage.md), [docs/configuration.md](docs/configuration.md), [docs/operations.md](docs/operations.md).
- Wire contract and SDK examples: [docs/api/](docs/api/README.md).
- Model pins, measurements and frontend limits: [docs/baseline.md](docs/baseline.md).
- Tests and image gate procedure/evidence: [docs/testing.md](docs/testing.md).
- Architecture: [docs/architecture.md](docs/architecture.md); release status, evidence and pending gates: [docs/roadmap.md](docs/roadmap.md).
