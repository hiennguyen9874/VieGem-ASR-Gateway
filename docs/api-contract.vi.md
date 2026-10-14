# API contract — QUAN-Transcribe gateway (profile OpenAI đã pin)

Thiết kế: [deployment-design.vi.md](deployment-design.vi.md). Baseline: [baseline.vi.md](baseline.vi.md).

Tài liệu này chốt payload cho R1 (REST/SSE) và R3a (Realtime explicit commit). Mọi ví dụ nằm trong
`tests/contract/examples/` và được **parse bằng SDK thật** (`tests/contract/test_python_sdk.py`,
`tests/contract/js/contract.test.ts`). Nếu tài liệu và ví dụ khác nhau thì ví dụ + test là chuẩn.

## 1. Profile đã pin

| | Pin | Kiểm chứng |
|---|---|---|
| OpenAI Python SDK | `openai==3.31.0` = commit `86cae50785fb7b7bf2a5d2383370b214b304acd2` (đúng commit thiết kế tham chiếu) | Client SDK thật qua mock transport: encoding request, parse JSON/text/SSE/error; Realtime event validate bằng pydantic union |
| OpenAI JS SDK | `openai@7.33.0` (gitHead `dd51b11156f920c57dc2e4ad4162cf28a42fe8a1`), TypeScript 5.9.3, Node ≥ 23.6 | `tsc --strict` với `satisfies` trên type SDK + client SDK thật với custom `fetch` |
| Realtime | GA schema (`session.type = "transcription"`), không dùng beta `transcription_session.update` | |

**Chưa kiểm chứng:** hành vi thật của server OpenAI (thứ tự event, mã lỗi chính xác). Contract dựa trên type SDK và
tài liệu OpenAI, không phải traffic quan sát được. Mã lỗi `code` bên dưới là của gateway này.

## 2. Capability matrix

| Tính năng | Trạng thái | Release |
|---|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | Hỗ trợ | R1 |
| `POST /v1/audio/transcriptions` `response_format=json\|text` | Hỗ trợ | R1 |
| `stream=true` (SSE delta/done) | Hỗ trợ | R1 |
| `verbose_json`, `srt`, `vtt`, `timestamp_granularities[]=segment` | 400 `unsupported_response_format`/`unsupported_parameter` cho tới R2 | R2 |
| `timestamp_granularities[]=word` | 400 cho tới khi qua alignment gate | R2 (riêng) |
| WS `/v1/realtime` transcription, `turn_detection: null`, append/commit/clear | Hỗ trợ | R3a |
| `server_vad` | 400 `unsupported_turn_detection` | R3b |
| `semantic_vad`, `noise_reduction`, `audio/pcmu`, `audio/pcma` | Từ chối | — |
| `prompt` không rỗng, `keywords`, `include[]=logprobs`, `diarized_json`, `known_speaker_*`, `chunking_strategy` | Từ chối 400 | — |
| Language khác `vi`, auto-detect, translations, speech, chat/responses, WebRTC, SIP, ephemeral client secrets | Không hỗ trợ | — |

Không có endpoint nào trả `logprobs`, `avg_logprob`, `no_speech_prob`, confidence hay token id kiểu Whisper.

## 3. Models

- Id: `quan-transcribe-1.0`. Alias `whisper-1` chỉ khi bật cấu hình (mặc định tắt); khi bật, `whisper-1` được chấp
  nhận ở mọi field `model` và được liệt kê trong `/v1/models` như một model riêng (cùng backend QUAN, output khác
  Whisper; ghi rõ trong README). Mọi id khác: 404 `model_not_found`.
- Object: `{"id", "object": "model", "created", "owned_by": "qnguyen3"}`. `created` = commit time của revision
  weights đã pin (`a08726d…` → `1791950390`). Ví dụ: `examples/models.json`, `examples/model.json`.

## 4. `POST /v1/audio/transcriptions`

Auth: `Authorization: Bearer <key>`. Body `multipart/form-data`.

| Field | Quy tắc |
|---|---|
| `file` | Bắt buộc, đúng một part. Tên file client bị bỏ qua (không dùng làm path, không log). Định dạng nhận diện theo nội dung, không theo đuôi: Container allowlist: WAV, FLAC, MP3/MPGA, MP4/M4A, OGG, WEBM/Matroska. Decoder allowlist: PCM (`s16le`, `s24le`, `s32le`, `u8`, `f32le`, `f64le`, A-law, µ-law), FLAC, ALAC, MP1/MP2/MP3, AAC, Vorbis, Opus. Chấp nhận mọi tổ hợp container/decoder trong hai allowlist mà FFmpeg đọc được (kể cả OGG/FLAC và Matroska/AAC); không giới hạn theo cặp. Container/codec ngoài allowlist (ADTS `.aac`, MPEG-PS, playlist, ADPCM, …) → 400 `invalid_audio`. Mọi sample rate/số kênh được đưa về mono 16 kHz (downmix của FFmpeg) |
| `model` | Bắt buộc; thiếu → 400 `missing_required_parameter`; không phục vụ → 404 `model_not_found` (`param: "model"`) |
| `language` | Không gửi hoặc `vi`; khác → 400 `unsupported_language`. Không gửi **không** có nghĩa là auto-detect |
| `temperature` | Không gửi hoặc số bằng 0 (`0`, `0.0`, `-0`; decode tất định, greedy). Không phải số hữu hạn (rỗng, `abc`, `nan`, `inf`) → 400 `invalid_value`; số khác 0 → 400 `unsupported_parameter` |
| `prompt` | Không gửi hoặc rỗng; khác → 400 `unsupported_parameter` |
| `response_format` | `json` (mặc định), `text`; R2 thêm `verbose_json`/`srt`/`vtt`; khác → 400 `unsupported_response_format` |
| `stream` | Đúng chuỗi `true` hoặc `false` (cả hai SDK gửi như vậy); giá trị khác (`True`, `1`, rỗng) → 400 `invalid_value`. `true` với `response_format` không gửi, `json` hoặc `text` → SSE (§5); R2: với format khác → 400 `unsupported_parameter` (`param: "stream"`; ở R1 format khác đã bị từ chối trước) |
| `include[]`, `timestamp_granularities[]` | Field lặp, tên **có** `[]` (đã xác minh cả hai SDK gửi như vậy). R1: bất kỳ giá trị nào → 400 `unsupported_parameter` |
| Field khác (kể cả `chunking_strategy`, `known_speaker_names[]`, `include` không có `[]`) | 400 `unsupported_parameter` (`param` = tên field, cắt 64 ký tự). Không âm thầm bỏ qua |

Mọi field trừ `file`, `include[]`, `timestamp_granularities[]` chỉ được gửi một lần; lặp → 400 `invalid_value`
(`param` = tên). Thứ tự kiểm tra cố định, trả lỗi đầu tiên (`app/transcription.py:parse_params`, test
`tests/unit/test_transcription_params.py`): (1) field lạ theo thứ tự gửi, (2) field lặp, (3) thiếu/rỗng `model`,
(4) `include[]`, (5) `timestamp_granularities[]`, (6) `language`, (7) `temperature`, (8) `prompt`,
(9) `response_format`, (10) `stream`, (11) model không phục vụ (404). Multipart/`file` được kiểm tra trước tất cả
(§4 giới hạn), decode audio sau tất cả.

Kết quả:

- `json` → `200 application/json`: `{"text": "...", "usage": {"type": "duration", "seconds": <float>}}`.
  `usage.seconds` = số mẫu audio sau decode / sample rate gốc (tính trước chunking), là đại lượng đo được.
  Không có `logprobs`/`languages`. Ví dụ `examples/transcription.json`.
- `text` → `200 text/plain; charset=utf-8`, body đúng bằng transcript, không thêm newline. SDK trả nguyên body.
- Text: chữ thường tiếng Việt, không dấu câu, số viết bằng chữ (output nguyên bản của model). Nhiều chunk nối bằng
  một dấu cách; chunk rỗng không sinh dấu cách.
- Silence/clip ngắn (chốt ở M2, `app/chunking.py`): audio decode hợp lệ nhưng ≤ 1,600 mẫu ở 16 kHz (≤ 100 ms), kể
  cả 0 mẫu → `text: ""`, không gọi model (frontend crash dưới 161 mẫu; xem baseline §7). Silence dài vẫn qua model
  và cho `""` (đã đo), không dùng ngưỡng biên độ.
- Chunking: audio dài hơn `QUAN_MAX_CHUNK_SECONDS` (mặc định 30 s) được cắt tại điểm yên lặng nhất trong 5 s cuối
  trước giới hạn; các chunk phủ liền toàn bộ audio (không bỏ silence, không bỏ đuôi), mỗi chunk > 100 ms. Text có
  thể khác một lời gọi `transcribe()` của reference trên cùng file vì điểm cắt khác.
- Giới hạn (cấu hình, xem `app/config.py`): toàn bộ body ≤ `QUAN_MAX_UPLOAD_BYTES` (25 MiB) → 413, kể cả upload
  chunked; ≤ 32 part, mỗi field không phải file ≤ 8 KiB (→ 413), field phải là UTF-8; audio sau decode ≤
  `QUAN_MAX_AUDIO_SECONDS` (1800 s) → 400 `audio_too_long`; decode quá `QUAN_DECODE_TIMEOUT_SECONDS` → 400
  `invalid_audio`; tổng upload/decode/queue/infer quá `QUAN_REQUEST_TIMEOUT_SECONDS` → 503 `request_timeout`; hết
  slot decode, spool disk, lane (`QUAN_QUEUE_MAX_LANES`, kiểm tra trước khi decode) hoặc budget audio đã decode giữ
  trong RAM (`QUAN_MAX_DECODED_SECONDS`, mặc định 3600 s cho mọi request đang chạy) → 429 `server_busy`. Stream bị
  cụt được decode phần đọc được nếu FFmpeg đọc được.

## 5. SSE (`stream=true`)

- Validate request **và decode audio xong trước khi gửi headers**; lỗi giai đoạn này trả HTTP lỗi như §7.
- Headers: `200`, `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, không có `Content-Length` (chunked). JSON trong event là compact UTF-8 (không escape
  `\uXXXX`). Không có heartbeat: reverse proxy phải tắt buffering và có read timeout ≥
  `QUAN_REQUEST_TIMEOUT_SECONDS` (README).
- Mỗi event là một dòng `data: <json>` + dòng trống; **không** có trường `event:` và **không** gửi `[DONE]`
  (cả hai SDK coi `[DONE]` là tùy chọn; stream kết thúc bằng đóng kết nối sau `done`).
- `{"type": "transcript.text.delta", "delta": "..."}`: một event cho mỗi chunk đã decode xong với text khác rỗng,
  theo đúng thứ tự audio. Delta thứ hai trở đi bắt đầu bằng dấu cách. Không gửi `logprobs`, `segment_id`.
- `{"type": "transcript.text.done", "text": "..."}` đúng một lần ở cuối; `"".join(deltas) == done.text`.
  Không gửi `usage` (type SDK của `done.usage` chỉ có biến thể `tokens`; gateway không có token count).
- Audio rỗng/ngắn: không có delta, chỉ một `done` với `text: ""`.
- Lỗi sau headers: một event `data: {"error": {"message", "type", "param", "code"}}` (cùng envelope với HTTP) rồi đóng
  stream, **không** có `done`. Cả hai SDK ném `APIError` với `code` tương ứng (đã test). Ví dụ
  `examples/transcription_stream.sse`, `examples/transcription_stream_error.sse`.
- Client ngắt kết nối: dừng enqueue chunk tiếp, hủy chunk đang chờ, bỏ kết quả chunk đang chạy; không phát gì
  thêm. Cũng áp dụng cho JSON/text và khi đang decode (FFmpeg bị kill). Client không đọc: một lần gửi event quá
  `QUAN_SEND_TIMEOUT_SECONDS` (10 s) được coi như ngắt kết nối (đóng kết nối, không gửi `error`).
- Tối đa `QUAN_QUEUE_MAX_JOBS_PER_LANE` (2) chunk của một request nằm trong queue/đang chạy cùng lúc: GPU chạy
  trước client tối đa một cửa sổ đó, server không buffer thêm event.

## 6. Realtime WebSocket `/v1/realtime` (R3a)

Kết nối: `GET /v1/realtime` (WebSocket), `Authorization: Bearer <key>` (server-to-server). Query `model` tùy chọn
(SDK `client.realtime.connect(model=…)` gửi `?model=`); nếu có thì phải là model phục vụ. Auth sai hoặc model sai →
từ chối handshake (401 / 404), không accept rồi mới báo lỗi. Không hỗ trợ subprotocol `openai-insecure-api-key.*`.

Session sau khi accept (server gửi `session.created` ngay):

```json
{"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
  "transcription": {"model": "quan-transcribe-1.0", "language": "vi"}, "turn_detection": null}}}
```

Client events:

| Event | Quy tắc |
|---|---|
| `session.update` | `session.type` phải là `transcription`. Field không gửi = giữ nguyên. Chấp nhận: `format` = `audio/pcm` rate 24000; `transcription.model` = model phục vụ; `transcription.language` = `vi`; `turn_detection: null`. Từ chối (event `error`, session không đổi): `pcmu`/`pcma`, `turn_detection` khác null, `noise_reduction` khác null, `prompt` không rỗng, `keywords`, `languages`, `delay`, `include`. Đổi format khi buffer còn audio → lỗi. Thành công → `session.updated` với session đầy đủ |
| `input_audio_buffer.append` | `audio` = base64 PCM16 little-endian mono 24 kHz. Base64 sai → `error` `invalid_audio`. Byte lẻ giữ lại cho packet sau. Vượt giới hạn buffer → `error` `input_audio_buffer_full` |
| `input_audio_buffer.commit` | Buffer 0 byte → `error` `input_audio_buffer_commit_empty`, không tạo item. Ngược lại tạo item mới, buffer mới rỗng ngay; quá số item đang chờ inference → `error` `too_many_pending_items` |
| `input_audio_buffer.clear` | Chỉ xóa pending audio, không ảnh hưởng item đã commit → `input_audio_buffer.cleared` |
| Loại khác (`response.create`, `conversation.item.create`, …) | `error` `unsupported_event`; type không tồn tại / JSON sai → `error` `invalid_event` |

Server events cho mỗi commit, theo thứ tự trong cùng item:

1. `input_audio_buffer.committed` `{event_id, item_id, previous_item_id}`
2. `conversation.item.added` `{event_id, previous_item_id, item}` với
   `item = {"id", "object": "realtime.item", "type": "message", "role": "user", "status": "completed", "content": [{"type": "input_audio"}]}`
3. 0..n `conversation.item.input_audio_transcription.delta` `{event_id, item_id, content_index: 0, delta}`: chỉ phát
   **sau commit**, một delta mỗi chunk nội bộ đã final; nối lại = transcript
4. Đúng một trong hai:
   `conversation.item.input_audio_transcription.completed` `{event_id, item_id, content_index: 0, transcript, usage: {"type": "duration", "seconds"}}`
   hoặc `conversation.item.input_audio_transcription.failed` `{event_id, item_id, content_index: 0, error: {type, code, message}}`

Quy tắc chung:

- `event_id` duy nhất trong session (`event_…`), `item_id` dạng `item_…`. `previous_item_id` = item commit trước
  đó trong cùng session hoặc `null`.
- `usage.seconds` = thời lượng audio đã commit (byte / 2 / 24000). Trường `usage` là **bắt buộc** trong type SDK
  của `completed`; không bao giờ phát `usage` dạng `tokens`.
- Completed của các item khác nhau có thể không theo thứ tự commit (OpenAI cũng không đảm bảo); client ghép bằng
  `item_id`. `completed.transcript` là kết quả authoritative.
- Không phát `conversation.item.done`, `input_audio_buffer.speech_started/stopped` (R3b), `response.*`,
  `rate_limits.updated`. Không có partial trước commit.
- `error` event: `{"type": "error", "event_id", "error": {"type", "code", "message", "param", "event_id": <event_id client gửi, nếu có>}}`.
- **Không gửi `null`** cho field optional mà type JS không cho null: bỏ hẳn `noise_reduction` và
  `content[].transcript` (JS type check bắt được hai lỗi này; Python SDK thì chấp nhận null).

Lý do chọn `conversation.item.added` thay vì `conversation.item.created` (thiết kế ghi `created`): cả hai đều có
trong union GA của SDK; docstring GA của `conversation.item.added` nêu rõ trường hợp "input audio buffer is committed",
còn `created` là event thời beta. Client chỉ cần transcript nên dựa vào `committed` + `completed`; nếu client thật
cần `created`, thêm sau bằng integration test.

Ví dụ: `examples/realtime_client.jsonl`, `examples/realtime_server.jsonl`.

## 7. Lỗi HTTP

Envelope (ví dụ `examples/error.json`):

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "language", "code": "unsupported_language"}}
```

| HTTP | `type` | `code` |
|---|---|---|
| 400 | `invalid_request_error` | `missing_required_parameter`, `unsupported_parameter`, `invalid_value` (kể cả multipart sai/quá nhiều field), `unsupported_language`, `unsupported_response_format`, `invalid_audio` (`param: "file"`: không decode được, định dạng không cho phép, decode quá thời gian, sample không hữu hạn), `audio_too_long` (`param: "file"`) |
| 401 | `authentication_error` | `invalid_api_key` (thiếu hoặc sai) |
| 404 | `invalid_request_error` | `model_not_found`, `not_found` (route không tồn tại; chỉ sau khi auth hợp lệ, nếu không là 401) |
| 405 | `invalid_request_error` | `method_not_allowed` |
| 413 | `invalid_request_error` | `request_too_large` (body vượt giới hạn, kiểm tra trước multipart parse; hoặc một form field vượt 8 KiB) |
| 429 | `rate_limit_error` | `server_busy` (queue/concurrency/decode slot/spool disk/budget audio đã decode đầy); kèm `Retry-After` khi có ước lượng |
| 500 | `server_error` | `inference_failed`, `internal_error` (lỗi không lường trước, không lộ chi tiết) |
| 503 | `server_error` | `model_not_ready` (đang load/recover, `Retry-After: 5`), `shutting_down` (đang drain), `request_timeout` (hết budget tổng của request ở upload, decode, chờ queue hoặc đang chạy; call đang chạy vẫn giữ worker/budget tới khi kết thúc) |

Mọi response (kể cả lỗi) có header `x-request-id: req_…`; Python SDK đưa giá trị này vào `request_id` của exception; JS SDK dùng `requestID`.

Endpoint vận hành (không thuộc profile OpenAI): `GET /health` (liveness, không cần auth; 503 `unhealthy` khi
cần restart process), `GET /ready` (readiness, không cần auth; 200 `{"status": "ready"}` hoặc 503 như trên),
`GET /metrics` (Prometheus text, cần API key).

Không bao giờ trả FastAPI 422 `{"detail": …}`. Message không chứa transcript, audio, token hay đường dẫn nội bộ.
SDK ánh xạ status → class (`BadRequestError`, `AuthenticationError`, `NotFoundError`, …) và đọc `code`/`param`/`type`
từ envelope. `examples/service_errors.json` được kiểm thử bằng cả Python và JS SDK cho các status/code M1,
`Retry-After` và request ID (M2 thêm `invalid_audio`, `audio_too_long`, 413 cho form field). API key phải có ít nhất 16 ký tự ASCII printable, không có khoảng trắng.

## 8. Kiểm tra contract

```bash
uv run pytest tests/contract tests/unit
(cd tests/contract/js && npm ci && npm test)
```

`tests/contract/test_server_sdk.py` chạy gateway thật (Uvicorn + FFmpeg, fake engine): payload có cùng key/type với
`examples/`, Python SDK parse được, và chạy `js/server.test.ts` (JS SDK) với `QUAN_TEST_BASE_URL`,
`QUAN_TEST_API_KEY`, `QUAN_TEST_AUDIO_DIR`; `npm test` đơn lẻ bỏ qua các test đó.

Khi đổi version SDK: cập nhật pin trong `pyproject.toml`/`tests/contract/js/package.json`, chạy lại cả hai, ghi lại
thay đổi schema ở tài liệu này. Khi gateway được triển khai (M3/M5), test tích hợp phải gọi server thật và so sánh
payload với các ví dụ ở đây.
