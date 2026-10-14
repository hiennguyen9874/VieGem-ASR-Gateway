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
| `file` | Bắt buộc. Tên file client chỉ để log/debug, không dùng làm path. Định dạng theo M2 (FFmpeg) |
| `model` | Bắt buộc; thiếu → 400 `missing_required_parameter`; không phục vụ → 404 `model_not_found` (`param: "model"`) |
| `language` | Không gửi hoặc `vi`; khác → 400 `unsupported_language`. Không gửi **không** có nghĩa là auto-detect |
| `temperature` | Không gửi hoặc `0` (decode tất định, greedy); khác → 400 `unsupported_parameter` |
| `prompt` | Không gửi hoặc rỗng; khác → 400 `unsupported_parameter` |
| `response_format` | `json` (mặc định), `text`; R2 thêm `verbose_json`/`srt`/`vtt`; khác → 400 `unsupported_response_format` |
| `stream` | `true`/`false`. `true` với `response_format` không gửi, `json` hoặc `text` → SSE (§5); với format khác → 400 `unsupported_parameter` (`param: "stream"`) |
| `include[]`, `timestamp_granularities[]` | Field lặp, tên **có** `[]` (đã xác minh cả hai SDK gửi như vậy). R1: bất kỳ giá trị nào → 400 `unsupported_parameter` |
| Field khác (kể cả `chunking_strategy`, `known_speaker_names[]`) | 400 `unsupported_parameter` (`param` = tên field). Không âm thầm bỏ qua |

Kết quả:

- `json` → `200 application/json`: `{"text": "...", "usage": {"type": "duration", "seconds": <float>}}`.
  `usage.seconds` = số mẫu audio sau decode / sample rate gốc (tính trước chunking), là đại lượng đo được.
  Không có `logprobs`/`languages`. Ví dụ `examples/transcription.json`.
- `text` → `200 text/plain; charset=utf-8`, body đúng bằng transcript, không thêm newline. SDK trả nguyên body.
- Text: chữ thường tiếng Việt, không dấu câu, số viết bằng chữ (output nguyên bản của model). Nhiều chunk nối bằng
  một dấu cách; chunk rỗng không sinh dấu cách.
- Silence/clip ngắn: audio decode hợp lệ nhưng ≤ 1,600 mẫu ở 16 kHz (≤ 100 ms), kể cả 0 mẫu → `text: ""`, không
  gọi model (frontend crash dưới 161 mẫu; xem baseline §7). Silence dài vẫn qua model và cho `""` (đã đo). Chốt ở M2.

## 5. SSE (`stream=true`)

- Validate request **và decode audio xong trước khi gửi headers**; lỗi giai đoạn này trả HTTP lỗi như §7.
- Headers: `200`, `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
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
- Client ngắt kết nối: dừng enqueue chunk tiếp; không phát gì thêm.

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
| 400 | `invalid_request_error` | `missing_required_parameter`, `unsupported_parameter`, `invalid_value`, `unsupported_language`, `unsupported_response_format`, `invalid_audio` (không decode được), `audio_too_long` |
| 401 | `authentication_error` | `invalid_api_key` (thiếu hoặc sai) |
| 404 | `invalid_request_error` | `model_not_found`, `not_found` (route không tồn tại; chỉ sau khi auth hợp lệ, nếu không là 401) |
| 405 | `invalid_request_error` | `method_not_allowed` |
| 413 | `invalid_request_error` | `request_too_large` (body/file vượt giới hạn, kiểm tra trước multipart parse) |
| 429 | `rate_limit_error` | `server_busy` (queue/concurrency đầy); kèm `Retry-After` khi có ước lượng |
| 500 | `server_error` | `inference_failed`, `internal_error` (lỗi không lường trước, không lộ chi tiết) |
| 503 | `server_error` | `model_not_ready` (đang load/recover, `Retry-After: 5`), `shutting_down` (đang drain), `request_timeout` (job quá hạn trước khi được chạy) |

Mọi response (kể cả lỗi) có header `x-request-id: req_…`; SDK đưa giá trị này vào `request_id` của exception.

Endpoint vận hành (không thuộc profile OpenAI): `GET /health` (liveness, không cần auth; 503 `unhealthy` khi
cần restart process), `GET /ready` (readiness, không cần auth; 200 `{"status": "ready"}` hoặc 503 như trên),
`GET /metrics` (Prometheus text, cần API key).

Không bao giờ trả FastAPI 422 `{"detail": …}`. Message không chứa transcript, audio, token hay đường dẫn nội bộ.
SDK ánh xạ status → class (`BadRequestError`, `AuthenticationError`, `NotFoundError`, …) và đọc `code`/`param`/`type`
từ envelope (đã test với 400).

## 8. Kiểm tra contract

```bash
uv run pytest tests/contract tests/unit
(cd tests/contract/js && npm ci && npm test)
```

Khi đổi version SDK: cập nhật pin trong `pyproject.toml`/`tests/contract/js/package.json`, chạy lại cả hai, ghi lại
thay đổi schema ở tài liệu này. Khi gateway được triển khai (M3/M5), test tích hợp phải gọi server thật và so sánh
payload với các ví dụ ở đây.
