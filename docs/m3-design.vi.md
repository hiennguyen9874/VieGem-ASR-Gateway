# M3 — Thiết kế và kế hoạch: REST/SSE và R1 container

Kế hoạch tổng: [implementation-plan.vi.md](implementation-plan.vi.md) §M3. Contract: [api-contract.vi.md](api-contract.vi.md)
§3–5, §7. Tài liệu này chốt cách nối các khối M1–M2 đã có thành endpoint và image R1.

**Trạng thái triển khai:** bước 1–5 và 7 xong; bước 6 đã viết file nhưng chưa build/chạy (D1). D2–D5 theo đề xuất
(D3: `QUAN_MAX_DECODED_SECONDS`). Khác với bản thiết kế, đã kiểm tra khi triển khai:
- `stop_grace_period: 70s` thay vì 45 s: Uvicorn chờ request đang mở (`--timeout-graceful-shutdown 30`) *trước* khi
  lifespan drain scheduler (30 s) và dừng worker (5 s), nên trường hợp xấu nhất là ~65 s.
- Smoke trong container gate chạy từ một sidecar container dùng chung network namespace (`--network
  container:<id>`, repo mount read-only) thay vì `docker exec`: image không chứa script/fixture và server chạy
  `--network none`.
- Send timeout khi SSE đã bắt đầu: handler kết thúc mà response chưa xong; theo code Uvicorn 0.54 (`h11_impl.py`,
  chưa quan sát qua socket thật) Uvicorn log "ASGI callable returned without completing response" rồi đóng kết nối.
  Log này là dấu hiệu client chậm, không phải lỗi server.

## 1. Hiện trạng (đã kiểm tra trong code)

| Khối | Đã có | M3 dùng thế nào |
|---|---|---|
| `app/main.py` | lifespan, `/health`, `/ready`, `/metrics`, `GatewayMiddleware` (auth Bearer, `x-request-id`, envelope 500 cuối cùng) | Đăng ký router của `app/transcription.py`; auth đã áp dụng cho mọi `/v1/*` |
| `app/audio.py` | `receive(request, deadline)` → `Upload(fields, path, size)` (fields giữ thứ tự, giữ tên lặp `include[]`); `decode(path, deadline)` → float32 16 kHz | Gọi đúng hai hàm này, không đọc body hay chạy FFmpeg ở chỗ khác |
| `app/chunking.py` | `split(wav, max_samples)` → `[Span]` phủ liền `[0, n)`; ≤ 1,600 mẫu → `[]` | Dùng chung cho JSON/text/SSE |
| `app/scheduler.py` | `open_lane(FILE)` (503/429 ngay), `lane.submit(wav, deadline)` (chờ capacity ≤ `queue_timeout`, 429), `Job` awaitable, `lane.close()` hủy job đang chờ và bỏ kết quả job đang chạy | Một lane cho mỗi request |
| `app/errors.py` | Toàn bộ mã lỗi §7 | Thêm vài constructor 400 nếu cần; không đổi mã đã có |
| `app/config.py` | `whisper_alias`, `request_timeout_seconds`, `send_timeout_seconds` (đang ghi chú cho M5) | Dùng `send_timeout_seconds` cho cả SSE; không thêm biến mới (trừ D3 nếu chọn) |
| `tests/contract/examples/` | models, json, text, SSE, SSE error | Server thật phải trả payload cùng shape |

Các chi tiết môi trường ảnh hưởng tới thiết kế:

- Uvicorn 0.54 khai báo ASGI `spec_version` "2.3"; Starlette 1.8 `StreamingResponse` chỉ chạy task nghe disconnect khi
  spec < 2.4. Nếu bản Uvicorn sau đổi thành 2.4, response chỉ phát hiện disconnect lúc `send` (Starlette bắt `OSError`),
  không phát hiện trong lúc chờ inference. Uvicorn cũng bỏ qua `send` sau disconnect mà không báo lỗi, và `send` chờ
  drain khi socket bị pause, không có giới hạn thời gian. → M3 tự viết watcher disconnect và tự giới hạn thời gian `send` (§4).
- `decode_units()` của reference đã `strip()`; chunk không có tiếng nói cho `""`.
- **Host này không có Docker/Podman** (`which docker podman` rỗng). Không build hoặc kiểm tra được container gate ở
  đây; xem §8.

## 2. Luồng request

```text
POST /v1/audio/transcriptions
 1. middleware: auth, request id                               (401)
 2. pre-check: scheduler.closing / not worker.ready            (503 shutting_down / model_not_ready, chưa đọc body)
 3. deadline = monotonic() + QUAN_REQUEST_TIMEOUT_SECONDS
 4. async with audio.receive(request, deadline) as upload:      (400/413/429/503, spool bị xóa khi thoát)
 5.   params = parse_params(upload.fields, config)              (400/404, thuần, không I/O)
 6.   lane = scheduler.open_lane(FILE)                          (429/503: từ chối trước khi tốn CPU FFmpeg)
 7.   wav = await audio.decode(upload.path, deadline)           (400/429/503)
 8.   (spool được xóa ở đây; giữ wav trong RAM)
 9.   spans = split(wav, max_chunk_samples)
10.   json/text: texts = run_chunks(...) → join → response
      stream:    gửi headers 200 SSE, rồi mỗi chunk xong → delta; cuối cùng → done
     finally: lane.close()
```

- Bước 2 chỉ để đỡ phải nhận 25 MB rồi mới báo 503; lane vẫn tự kiểm tra lại ở bước 6.
- Lane được mở **trước decode**. Lane được giữ trong suốt quá trình decode nên `QUAN_QUEUE_MAX_LANES` giới hạn luôn
  số request đang ở bước decode/inference; request bị từ chối ngay mà không phải tốn FFmpeg.
- Spool được xóa ngay sau decode (thoát context `receive` trước bước inference); không giữ file tạm trong lúc chờ GPU.
- Bước 4–9 chạy trong watcher disconnect (§4): client ngắt kết nối thì task bị hủy, FFmpeg bị kill (M2 đã xử lý),
  lane bị đóng.

## 3. `app/transcription.py`

### 3.1 Models

- `GET /v1/models` → `{"object": "list", "data": [model…]}`; `GET /v1/models/{model_id:path}` → model hoặc 404
  `model_not_found` (`:path` để id có `/` cũng trả `model_not_found`, không trả `not_found`).
- Danh sách: `quan-transcribe-1.0`, thêm `whisper-1` nếu `QUAN_WHISPER_ALIAS=true`. `created = 1791950390`,
  `owned_by = "qnguyen3"`, khai báo hằng số trong `app/transcription.py`. Thêm một unit test kiểm tra hằng số này khớp
  `examples/model.json`.

### 3.2 Validate form (hàm thuần `parse_params(fields, config) -> Params`)

`Params = (model, response_format: "json"|"text", stream: bool)`. Kiểm tra theo **thứ tự cố định**, trả lỗi đầu tiên
gặp được (có test cho thứ tự này):

| # | Điều kiện | Lỗi |
|---|---|---|
| 1 | Tên field ngoài `{model, language, temperature, prompt, response_format, stream, include[], timestamp_granularities[]}` (theo thứ tự field gửi lên) | 400 `unsupported_parameter`, `param` = tên field (cắt 64 ký tự) |
| 2 | Field đơn (mọi field trừ hai field `[]`) xuất hiện nhiều lần | 400 `invalid_value`, `param` = tên |
| 3 | Thiếu `model` / rỗng | 400 `missing_required_parameter`, `param: "model"` |
| 4 | `include[]` có mặt (bất kỳ giá trị nào) | 400 `unsupported_parameter`, `param: "include[]"` |
| 5 | `timestamp_granularities[]` có mặt | 400 `unsupported_parameter`, `param: "timestamp_granularities[]"` |
| 6 | `language` khác `vi` (kể cả rỗng) | 400 `unsupported_language` |
| 7 | `temperature`: không parse được float hữu hạn → `invalid_value`; khác 0 → `unsupported_parameter` | 400, `param: "temperature"` |
| 8 | `prompt` không rỗng | 400 `unsupported_parameter` |
| 9 | `response_format` ∉ {json, text}; `verbose_json`/`srt`/`vtt`/`diarized_json` cũng bị từ chối ở R1 | 400 `unsupported_response_format` |
| 10 | `stream` ∉ {`true`, `false`} | 400 `invalid_value`, `param: "stream"` |
| 11 | `model` không phục vụ | 404 `model_not_found` |

Các lỗi 400 được kiểm tra trước 404 để client nhận lỗi cú pháp trước. `stream=true` + json/text → SSE. Ở R1, nhánh
"stream + format khác" (§4 contract) đã bị bước 9 chặn trước, nên không xảy ra; R2 sẽ cần nhánh này.

Hai SDK gửi `temperature` là `"0"` và `stream` là `"true"` (`tests/contract/test_python_sdk.py`). Nếu chấp nhận
thêm `"0.0"` thì không có rủi ro gì.

### 3.3 Chạy chunk và nối text (dùng chung cho mọi format)

```python
async def run_chunks(lane, wav, spans, deadline) -> AsyncIterator[str]:
    """Yield chunk texts in audio order; at most `queue_max_jobs_per_lane` jobs in flight."""
```

- Cửa sổ pipeline: submit chunk tiếp theo trước khi chờ chunk cũ nhất, tối đa `queue_max_jobs_per_lane` job (mặc định
  2). GPU không phải chờ client nhận event, và không bao giờ enqueue hết chunk của file dài.
- `submit(wav[s.start:s.end], deadline)`: slice là view, không copy. Toàn bộ vòng lặp chạy trong
  `asyncio.timeout_at(deadline)`; hết hạn thì 503 `request_timeout` (job đang chạy vẫn giữ worker, đúng như M1).
- `Joiner`: bỏ chunk rỗng; delta của chunk khác rỗng = text, thêm `" "` ở đầu nếu đã có chunk khác rỗng trước đó.
  `done.text`/`json.text`/`text` body = `"".join(deltas)`. Đây là đoạn code duy nhất làm việc nối text.
- `usage.seconds = len(wav) / 16000` (không làm tròn); chỉ có trong JSON, không có trong SSE done.

### 3.4 Response

- `json`: `JSONResponse({"text", "usage": {"type": "duration", "seconds"}})`, không có field nào khác.
- `text`: `Response(text, media_type="text/plain; charset=utf-8")`, không thêm newline.
- Audio rỗng/≤ 100 ms: `spans == []` → `""`, không submit job; SSE chỉ gửi một `done`.

## 4. SSE và disconnect

ASGI helper nhỏ (`app/transcription.py` hoặc `app/sse.py`), không dùng `StreamingResponse`:

- `watch_disconnect(receive, task)`: lúc này body đã được đọc hết, nên `receive()` chỉ còn có thể trả
  `http.disconnect`; nhận được thì hủy task đang xử lý (decode/inference/stream). Dùng cho cả JSON/text (bước 4–10)
  lẫn SSE.
- Headers: `200`, `content-type: text/event-stream; charset=utf-8`, `cache-control: no-cache`,
  `x-accel-buffering: no`; không có `content-length`. Chỉ gửi headers sau khi decode thành công (contract §5).
- Mỗi event `data: <json compact, ensure_ascii=False>\n\n`. Mỗi `send` nằm trong
  `asyncio.timeout(send_timeout_seconds)`; hết hạn thì coi như disconnect (đóng lane, log `outcome=send_timeout`).
  Producer chờ `send` xong mới lấy kết quả tiếp, nên outbound buffer ≤ 1 event + buffer transport; GPU vẫn chạy trước
  tối đa một cửa sổ chunk.
- Lỗi sau headers: `APIError` → `data: {"error": {...}}` rồi `more_body=False`; exception khác → log + `internal_error`
  event. Không bao giờ gửi `done` sau lỗi. Các lỗi có thể xảy ra ở đây: `inference_failed`, `request_timeout`,
  `server_busy` (submit chờ capacity quá hạn), `model_not_ready` (worker FAILED), `shutting_down`.
- Disconnect: không gửi gì thêm, `lane.close()` (job đang chờ bị hủy, kết quả job đang chạy bị bỏ).
- Không có heartbeat comment ở R1. Proxy timeout phải ≥ `QUAN_REQUEST_TIMEOUT_SECONDS` (ghi trong README).

## 5. Quan sát

- Log `transcription.request`: `response_format`, `stream`, `audio_s`, `chunks`, `outcome`, `duration_ms` (không ghi
  text, filename hay giá trị field).
- Metrics: `transcriptions_total{response_format,stream,outcome}` với outcome ∈ ok, error code, `disconnected`,
  `send_timeout`; `transcription_audio_seconds_total`; histogram `sse_first_event_seconds` (tính từ lúc nhận request).
  Số đo cho M7.

## 6. Container R1

| File | Quyết định |
|---|---|
| `Dockerfile` | Multi-stage. Base `ubuntu:24.04` pin digest: FFmpeg 6.1.1 cùng version với FFmpeg mà test M2 đã chạy (allowlist decoder theo tên). Builder: `uv` (image `ghcr.io/astral-sh/uv:<version>` pin), `git`, `uv python install 3.12.14` (version của baseline), `uv sync --frozen --no-dev`. Runtime: `ffmpeg` (apt, pin version), copy `/opt/python` + `.venv` + `app/` + `assets/manifest.json`. Wheel torch cu126 đã chứa thư viện CUDA, không cần base `nvidia/cuda`. User `quan` (uid 10001), `WORKDIR /srv`, `EXPOSE 8000`, `HEALTHCHECK` gọi `/health` bằng `python -c urllib`. `CMD uvicorn --factory app.main:create_app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log --timeout-graceful-shutdown 30` |
| `.dockerignore` | Allowlist: `*` rồi `!app/`, `!assets/manifest.json`, `!pyproject.toml`, `!uv.lock`; loại `__pycache__`. Không bao giờ có `references/`, `models/`, `fixtures*`, `tests/`, `.venv`, `.git`, `.env` |
| `compose.yaml` | Service `quan`: GPU qua `deploy.resources.reservations.devices` (`device_ids: ["${QUAN_GPU:-0}"]`); `./models:/models:ro`; `QUAN_MODELS_DIR=/models`, `QUAN_DEVICE=cuda`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/tmp/hf`, `HOME=/tmp`; `QUAN_API_KEYS_FILE=/run/secrets/quan_api_keys` (compose secret từ file, không đưa vào env); `read_only: true` + `tmpfs: /tmp` (size ≥ `QUAN_MAX_SPOOL_BYTES` + chỗ cho cache); `stop_grace_period: 45s` (> drain 30 s + worker 5 s); `restart: unless-stopped`; port chỉ bind `127.0.0.1:8000` (TLS/public qua reverse proxy). Profile `stage`: dùng cùng image chạy `scripts/stage_models.py` với network + `HF_TOKEN` runtime (hoặc stage trên host bằng `uv`) |

Ghi chú:
- Staging script cần có trong image thì phải copy thêm `scripts/stage_models.py`. Nếu chọn chỉ stage trên host thì
  bỏ profile `stage`. Đề xuất: stage trên host cho R1, ít thứ phải kiểm tra hơn.
- `/health` 503 (treo hoặc lỗi load) không làm Docker restart container: Docker chỉ restart khi process thoát. Xem D2.

## 7. Test và gate

| Lớp | File | Nội dung chính |
|---|---|---|
| Unit (CPU) | `tests/unit/test_transcription_params.py` | Bảng §3.2: mỗi hàng có ca hợp lệ và ca lỗi; thứ tự lỗi; field lặp; alias bật/tắt; `Joiner` (chunk rỗng ở đầu/giữa/cuối, `"".join(deltas) == done`) |
| E2E CPU (Uvicorn thật + FFmpeg thật + `FakeEngine`) | `tests/unit/test_transcription_api.py` | Python SDK thật gọi models/json/text/stream; audio nhiều chunk (`max_chunk_seconds=1`, khoảng 3.5 s) cho delta đúng thứ tự; clip ≤ 100 ms và 0 mẫu → `""` mà fake engine không bị gọi; 400/404/413/429/503 có envelope; `engine.max_active == 1` khi có nhiều request song song; hai request đồng thời không lẫn text của nhau; lỗi inference giữa stream → `APIError`, không có done; deadline giữa stream → event `request_timeout`; ngắt kết nối khi đang chờ inference → không còn job nào được submit (đếm `engine.calls`), lane được đóng, spool rỗng; ngắt kết nối khi đang decode → FFmpeg được reap; drain lúc shutdown trong khi stream đang chạy |
| ASGI (CPU) | cùng file | Send timeout: gọi app trực tiếp với `send` bị block → lane đóng, log `send_timeout` (kiểm tra qua socket thật không ổn định vì event nhỏ hơn buffer kernel) |
| Contract | `tests/contract/test_server_sdk.py`, `tests/contract/js/server.test.ts` | Payload server thật có cùng tập key/type với `examples/`. JS: pytest khởi server fake rồi chạy `node --test server.test.ts` với `QUAN_TEST_BASE_URL`/`QUAN_TEST_API_KEY`; skip nếu không có node hoặc `node_modules`; `npm test` tự skip khi chưa có env |
| GPU | `tests/gpu/test_server.py` | App thật trên GPU còn trống: `vi_a` json == text baseline offline; stream join == done == json; `vi_a` ‖ `vi_b` đồng thời không lẫn text; `vi_long.m4a` có WER nằm trong ngưỡng M2; silence → `""` |
| Smoke (artifact) | `scripts/smoke_server.py --base-url … --out docs/baseline/server[-container].json` | Dùng chung cho local và container: `/ready`, models, json/text/stream trên các fixture, text so với baseline, thời gian; ghi JSON có version/commit |
| Container | `tests/container/test_image.py` (skip nếu không có `docker`) | Build; `id -u` ≠ 0; `docker inspect`/`history` không có `HF_TOKEN`/key; image không chứa `references`/`models`/`tests`; `--network none` + model mount ro → `/ready` rồi chạy smoke qua `docker exec`; SIGTERM khi đang stream → thoát trong grace period, không mất response đang chạy dở trong lúc drain; restart xong `/ready` lại |

Cần FakeEngine trả text xác định cho audio thật (đã decode bằng FFmpeg): thêm hook `text_fn` tùy chọn (mặc định vẫn
`str(int(wav[0]))`), không đổi các test M1/M2 đang có.

**Gate R1** (giữ nguyên plan): hai SDK parse được JSON + SSE từ server thật; container offline có `/ready` sau
warm-up; test auth/error/overload pass; test isolation trên GPU pass; smoke so với baseline (cùng chunk và decoder
settings).

## 8. Kế hoạch theo bước

| Bước | Việc | Hoàn tất khi |
|---|---|---|
| 1 | `parse_params`, `Joiner`, models endpoints, hook `text_fn` cho FakeEngine | Unit test §3.2 pass; SDK `models.list/retrieve` chạy với server thật |
| 2 | Endpoint json/text: luồng §2, `run_chunks`, watcher disconnect, log/metrics | E2E CPU cho json/text, lỗi, isolation, disconnect pass; không còn temp file hay subprocess |
| 3 | SSE: ASGI response, send timeout, error event | Test stream, lỗi giữa stream, deadline, disconnect, send timeout pass; `uv run pytest` sạch |
| 4 | Contract với server thật (Python + JS); cập nhật `api-contract.vi.md` (thứ tự validate, field lặp, giá trị hợp lệ của `stream`/`temperature`, header SSE, send timeout, proxy timeout) | Cả hai SDK pass; example và tài liệu khớp nhau |
| 5 | Test GPU `test_server.py` + `scripts/smoke_server.py` → `docs/baseline/server.json` | Pass trên GPU còn trống; artifact đã commit |
| 6 | `Dockerfile`, `.dockerignore`, `compose.yaml`, `tests/container/` | **Cần host có Docker + NVIDIA Container Toolkit**: build, offline `/ready`, smoke → `docs/baseline/server-container.json` |
| 7 | `README.md` (staging, chạy offline, curl/SDK, capability, limits, proxy/TLS: `proxy_buffering off`, read timeout ≥ request timeout, body limit ≥ `QUAN_MAX_UPLOAD_BYTES`; recovery/restart); tick M3 trong plan; cập nhật AGENTS.md (cấu trúc mới, lệnh) | Review diff; mọi lệnh trong README đã chạy thử, trừ phần Docker nếu bước 6 bị chặn |

Bước 1–5 làm được trên host này. Bước 6 mới viết được file, chưa kiểm chứng được cho tới khi có host Docker.

## 9. Quyết định cần chốt (đề xuất mặc định)

| # | Quyết định | Đề xuất |
|---|---|---|
| D1 | Nơi verify container gate | Cần một host có Docker + NVIDIA Container Toolkit (host này không có). Nếu chưa có thì R1 gate để **chưa đạt** ở phần container, không đánh dấu hoàn thành |
| D2 | Restart khi `/health` 503 | R1: README hướng dẫn supervisor (systemd/k8s liveness hoặc autoheal). Watchdog tự `exit` khi worker treo để sang M7 |
| D3 | RAM của audio đã decode | Mỗi request giữ `wav` trong RAM tới khi xong: tối đa `queue_max_lanes × max_audio_seconds × 64 KB/s` (mặc định 32 × 1800 s ≈ 3.7 GB). Đề xuất M3 thêm budget tổng `QUAN_MAX_DECODED_SECONDS` (mặc định 3600 s ≈ 230 MB), reserve sau decode, vượt budget → 429. Phương án khác: giữ nguyên và ghi giới hạn vào README, để M7 đo |
| D4 | Base image | `ubuntu:24.04` + uv-managed Python 3.12.14 (cùng FFmpeg/Python với baseline). Base `python:3.12-slim` (Debian) có FFmpeg 5.1, khác bản mà allowlist đã kiểm tra |
| D5 | Staging | Trên host (`uv run scripts/stage_models.py`); image không chứa script staging hay `HF_TOKEN` |
