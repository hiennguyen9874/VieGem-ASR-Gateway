# Change rules by area

Read the section for each area you touch before editing it. File roles: [code-map.md](code-map.md).

## Inference path (`engine.py`, `scheduler.py`, `viegem_loader.py`)

- Load the Model only through `app/viegem_loader.load_viegem`: staged local files, weights-only checkpoint, no Hub
  id resolution.
- `EngineWorker` is the single thread that calls the Model (`CTCModel._acts` is mutable hook state). Work reaches it as
  jobs (one per chunk) on `Scheduler` lanes (one per request or realtime session).
- A lane holds at most `VIEGEM_QUEUE_MAX_JOBS_PER_LANE` (2) queued chunks; long files are fed incrementally. LIVE
  lanes get bounded priority (`live_priority_ratio`).
- Cancel drops queued chunks and discards the running chunk's result. A running kernel finishes; the worker and
  budgets are released only when the real call returns.
- LM (`app/lm.py`, ADR 0002): the engine thread never runs beam search; it only returns the pruned `Lattice`. The
  LM pool is a bounded spawn process pool (`VIEGEM_LM_WORKERS`, `VIEGEM_LM_MAX_PENDING`); a cancelled decode keeps
  its slot until the process finishes. Decoding constants are the model card's (not operator settings).
  `app/viegem/lm_decode.py` must keep matching upstream's search (`tests/unit/test_lm_decode.py`).

## Vendored VieGem inference (`app/viegem/`)

- `app/viegem/` is the CC-BY-NC-SA-4.0 licence boundary; retain its LICENSE and update NOTICE when changing it.
- Keep `references/VieGem-ASR` read-only and pinned to the manifest checkpoint revision. After changes, run
  `tests/gpu/test_viegem_parity.py` against staged assets on a free GPU; exact logits must match upstream
  for full-context and streaming inference, at fp32 (serving) and bf16 autocast. Preserve top-k encoder
  `requires_grad` flags.

## Audio (`audio.py`, `chunking.py`)

- Decode client audio only with `AudioIngress.decode`; chunk only with `chunking.split`.
- FFmpeg gets fixed argv, the `file` protocol and the demuxer/decoder allowlists; client filenames, paths and URLs
  stay out of its arguments, out of disk paths and out of logs.
- The frontend crashes below 161 samples at 16 kHz. Audio <= 100 ms (1,600 samples) yields `""` without a model
  call (docs/baseline.vi.md §5). After inference, all-zero chunks yield empty text/path with the measured frame
  count retained. Never use an unmeasured amplitude threshold to suppress nonzero audio.
- Every resource is budgeted: body limit before the multipart parser, spool bytes, decoded audio in RAM (`hold`,
  realtime pending audio via `reserve`). Exceeding one returns 413/429/503 in the OpenAI envelope.

## Timestamps (`alignment.py`, `formats.py`)

- Times come only from the CTC path of the same inference that produced the text (greedy path, or with the LM the
  forced alignment of its units on the same lattice), plus the chunk offset. On mismatch raise `AlignmentError`
  -> 500 `alignment_failed`.
- `WORD_TIMESTAMPS = False` in `app/transcription.py` until the R2b gate passes (docs/roadmap.md §2); flip it in the
  commit that carries the gate report.

## Realtime (`realtime.py`)

- Run Uvicorn with `--ws websockets-sansio --ws-max-size 2097152`; `--ws auto` silently disables WebSockets.
- Auth, model and capacity checks happen before `accept`.
- One LIVE lane per session; committed items run FIFO through the same `split` / `run_chunks` / `Joiner` as REST.
- The JS realtime SDK connects only over `wss:`; tests use a self-signed cert via `serve(tls_dir=...)`.

## API contract (`transcription.py`, `realtime.py`, `errors.py`)

- `tests/contract/examples/` plus the SDK tests are the source of truth; docs/api-contract.vi.md describes them.
- To change the wire format, update in this order: the examples, `tests/contract/test_python_sdk.py`,
  `tests/contract/js/contract.test.ts` (and `server.test.ts` when server behaviour changes), then the contract doc
  and capability matrix.
- Omit optional fields instead of sending `null`; the JS types reject `null` for several of them.
- Example files are byte-exact (SRT/VTT/SSE end with a blank line, `transcription.txt` has no trailing newline);
  pre-commit excludes them, so keep editors from normalising them too.

## Config (`config.py`)

- Every `VIEGEM_*` variable is a server setting, and unknown ones abort startup. Legacy `QUAN_*` variables
  also abort startup. `VIEGEM_GPU`, `VIEGEM_BUILD_CUDA` (compose) and `VIEGEM_TEST_*` (JS server test) are
  therefore kept out of the server's environment.
- Each limit is finite and validated at startup. A new setting also gets a row in the README config table.

## Pins, dependencies, image

- Change a model pin only in `assets/manifest.json`, then run `scripts/stage_models.py --update-manifest`. The
  vendored `app/viegem/` code and `references/VieGem-ASR` stay at the manifest's `viegem-asr` revision
  (`tests/gpu/test_viegem_parity.py`).
- CUDA profiles are mutually exclusive extras with identical model/code pins: cu126 = torch 2.14.1, cu128 = torch
  2.11.0 (the latest published cu128 wheel). Profile selectors: Docker `--build-arg TORCH_CUDA=cu128`, Compose
  `VIEGEM_BUILD_CUDA=cu128`, tests `--container-cuda=cu128`.
- The image contains only `app/` and the venv; models are mounted read-only, keys come from a Docker secret.
- After changing torch, CUDA, the base image or FFmpeg pins: rerun unit + contract tests and the full container
  gate for the target profile with no skips (docs/container-gate.vi.md).
- `docs/baseline/*.json` reports hold metadata only. CPU-diagnostic container reports stay uncommitted.
