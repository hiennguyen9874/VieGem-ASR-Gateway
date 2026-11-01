# Inference and scheduling

Read the relevant [ADRs](../adr/) before changing these boundaries; the code and test locations are in [code-map.md](code-map.md).

- Load the model through `app/viegem_loader.load_viegem` from staged local files with a weights-only checkpoint; serving must not resolve Hub IDs.
- `EngineWorker` is the only thread calling the model: `CTCModel._acts` is mutable hook state. Endpoints submit chunk jobs via `Scheduler` lanes and use `run_chunks` + `Joiner`; a lane represents a request or Realtime session.
- Keep per-lane admission bounded and incremental, with bounded priority for LIVE lanes. Cancellation drops queued work and discards the running result; the worker and associated budgets remain held until the actual call returns.
- LM beam search runs in the bounded CPU process pool, never on the engine thread. The engine returns a pruned lattice; cancelled decode retains its pool slot until it finishes. Keep model-card decoding constants and upstream search parity (`tests/unit/test_lm_decode.py`); see [ADR 0002](../adr/0002-lm-decoding-off-the-engine-thread.md).

## Vendored code

`app/viegem/` is the CC-BY-NC-SA-4.0 license boundary. Preserve its LICENSE and update NOTICE when changing vendored code. Keep the read-only `references/VieGem-ASR` revision aligned with the manifest. Run `tests/gpu/test_viegem_parity.py` with staged assets on a free GPU after changes: full-context and streaming logits must match upstream exactly at fp32 and bf16; preserve top-k encoder `requires_grad` flags. Report explicitly if parity cannot be run.
