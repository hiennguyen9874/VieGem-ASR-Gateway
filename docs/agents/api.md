# API, Realtime and configuration

## Wire contract

`tests/contract/examples/` and SDK tests are authoritative; [API contract](../api/README.md) describes their payloads, validation order and errors. For a wire change, update the examples, Python and JS SDK tests, server SDK tests when server behavior changes, and the contract and capability matrix. Run both SDK suites and the applicable server tests. Omit optional fields where JS types reject `null`.

Example bytes matter: SRT/VTT/SSE end in a blank line and `transcription.txt` has no trailing newline. Preserve these bytes even when editors normalize files.

## Realtime

Check auth, model and capacity before WebSocket accept. Each session owns one LIVE lane; committed items run FIFO through the REST `split` / `run_chunks` / `Joiner` path. Run Uvicorn with `--ws websockets-sansio --ws-max-size 2097152`: `--ws auto` silently disables WebSockets. The JS Realtime SDK requires `wss:`; tests use a self-signed certificate via `serve(tls_dir=...)`.

## Configuration

All `VIEGEM_*` environment variables reaching the server are settings; unknown names and legacy `QUAN_*` names abort startup. Keep Compose selectors (`VIEGEM_GPU`, `VIEGEM_BUILD_CUDA`) and JS test variables (`VIEGEM_TEST_*`) out of the server environment. New limits must be finite and validated at startup; document new settings in the README configuration table.
