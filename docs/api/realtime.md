# Realtime WebSocket `/v1/realtime`

Transcription qua WebSocket, commit tường minh (`turn_detection: null`). Ví dụ: `examples/realtime_client.jsonl`, `examples/realtime_server.jsonl`.

## Kết nối

`GET /v1/realtime`, header `Authorization: Bearer <key>` (server-to-server). Query tùy chọn, mỗi tham số tối đa một lần:

- `model`: model phục vụ (Python SDK `client.realtime.connect(model=…)`);
- `intent=transcription` (JS SDK `OpenAIRealtimeWS({intent: "transcription"})`).

Lỗi handshake là HTTP response **trước accept** (envelope [errors.md](errors.md)), theo thứ tự:

| Điều kiện | Kết quả |
|---|---|
| Auth sai | 401 |
| Tham số query khác (kể cả `call_id`) | 400 `unsupported_parameter` |
| Tham số lặp, `intent` khác | 400 `invalid_value` |
| Model không phục vụ | 404 `model_not_found` |
| Đang drain / chưa ready | 503 `shutting_down` / `model_not_ready` |
| Đủ `VIEGEM_MAX_SESSIONS` hoặc hết lane | 429 `server_busy` (`Retry-After: 1`) |

- JS SDK luôn dùng `wss:`: cần TLS ở hạ tầng phía trước.
- Message client tối đa 2 MiB (≈ 32 s audio); lớn hơn → đóng 1009. Message binary → `error` `invalid_event`.
- Không hỗ trợ subprotocol `openai-insecure-api-key.*`.

Sau accept, server gửi `session.created`:

```json
{"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
  "transcription": {"model": "viegem-asr", "language": "vi"}, "turn_detection": null}}}
```

## Client events

**`session.update`**: field vắng hoặc `null` = giữ nguyên. Thành công → `session.updated` với session đầy đủ. Lỗi → `error`, session **không đổi gì**; `param` = đường dẫn chấm (vd. `session.audio.input.turn_detection`).

| Giá trị | Kết quả |
|---|---|
| `session.type=transcription` (bắt buộc), format `audio/pcm` 24000, model phục vụ, `language=vi`, `prompt` rỗng, `turn_detection`/`noise_reduction` = `null` | Chấp nhận |
| Key lạ | `unknown_parameter` |
| Sai kiểu, `session.type`/format type/rate khác | `invalid_value` |
| `audio/pcmu`, `audio/pcma`, `noise_reduction`, `prompt` không rỗng, `keywords`, `languages`, `delay`, `include` | `unsupported_parameter` |
| `turn_detection` khác null | `unsupported_turn_detection` |
| Language khác / model không phục vụ | `unsupported_language` / `model_not_found` |

**`input_audio_buffer.append`**: `audio` = base64 PCM16 little-endian mono 24 kHz. Thành công không có event.

- Thiếu `audio` → `missing_required_parameter`; không phải chuỗi → `invalid_value`; base64 sai → `invalid_audio`.
- Buffer > `VIEGEM_MAX_TURN_SECONDS` (120 s) → `input_audio_buffer_full`; hết budget RAM toàn server → `server_busy`.
- Mọi lỗi: **toàn bộ** append bị bỏ, buffer không đổi. Byte lẻ giữ cho packet sau.

**`input_audio_buffer.commit`**:

- Buffer rỗng (0 mẫu) → `input_audio_buffer_commit_empty`, buffer giữ nguyên.
- Đã có `VIEGEM_MAX_PENDING_ITEMS` (4) item chưa xong → `too_many_pending_items`, buffer **giữ nguyên**.
- Ngược lại: byte lẻ cuối bị bỏ, tạo item mới, buffer rỗng ngay (nhận turn kế tiếp trong lúc item trước infer).
- Turn ≤ 100 ms → `completed` với `transcript: ""`, không gọi model.

**`input_audio_buffer.clear`**: xóa pending audio, không ảnh hưởng item đã commit → `input_audio_buffer.cleared`.

**Khác:** type GA khác (`response.create`, `conversation.item.create`, …) → `unsupported_event`; JSON sai, thiếu `type`, type ngoài union GA (kể cả beta `transcription_session.update`) → `invalid_event`; field lạ → `unknown_parameter`; `event_id` không phải chuỗi → `invalid_value`. Lỗi event không đóng socket.

## Audio

PCM16 / 32768 → float32, resample 24 → 16 kHz bằng soxr. Waveform của turn bằng tuyệt đối soxr one-shot trên cả turn, độc lập cách chia packet. Text có thể khác REST trên cùng file (resampler, quantize PCM16, ranh giới turn). Turn > 30 s được chia chunk như REST: nhiều delta, một item.

## Server events cho mỗi commit

1. `input_audio_buffer.committed` `{event_id, item_id, previous_item_id}`
2. `conversation.item.added` `{event_id, previous_item_id, item}`, `item = {"id", "object": "realtime.item", "type": "message", "role": "user", "status": "completed", "content": [{"type": "input_audio"}]}`
3. 0..n `conversation.item.input_audio_transcription.delta` `{event_id, item_id, content_index: 0, delta}`, một mỗi chunk; nối lại = transcript
4. Đúng một trong hai:
   - `...transcription.completed` `{event_id, item_id, content_index: 0, transcript, usage: {"type": "duration", "seconds"}}`
   - `...transcription.failed` `{event_id, item_id, content_index: 0, error: {type, code, message}}`

## Quy tắc chung

- `event_id` duy nhất trong session (`event_…`), `item_id` dạng `item_…`; `previous_item_id` = item commit trước hoặc `null`.
- `usage.seconds` = byte / 2 / 24000. Không bao giờ phát `usage` dạng `tokens`.
- Item xử lý FIFO theo thứ tự commit; event của item trước có thể xen giữa `committed`/`added` của item sau. Client nên ghép theo `item_id`. `completed.transcript` là kết quả chuẩn.
- Deadline mỗi item = lúc commit + `VIEGEM_REQUEST_TIMEOUT_SECONDS`. `failed.error.code`: `inference_failed`, `request_timeout`, `server_busy`, `model_not_ready`, `shutting_down`, `internal_error`. Session tiếp tục sau `failed`.
- Đóng session:
  - không hoạt động quá `VIEGEM_SESSION_IDLE_TIMEOUT_SECONDS` → `error` `idle_timeout`, close 1000;
  - quá `VIEGEM_MAX_SESSION_SECONDS` → `error` `session_expired`, close 1000;
  - client ngắt hoặc không đọc quá `VIEGEM_SEND_TIMEOUT_SECONDS` → kết thúc không gửi gì;
  - server shutdown → close 1012 ngay, item chưa xong bị mất.
- Không phát: `conversation.item.done`, `speech_started/stopped`, `response.*`, `rate_limits.updated`, partial trước commit.
- `error` event: `{"type": "error", "event_id", "error": {"type", "code", "message", "param", "event_id"}}`. `error.event_id` chỉ có khi event client gây lỗi có `event_id` là chuỗi printable ≤ 64 ký tự.
- Mã `error`: `invalid_event`, `unsupported_event`, `unknown_parameter`, `missing_required_parameter`, `invalid_value`, `unsupported_parameter`, `unsupported_turn_detection`, `unsupported_language`, `model_not_found`, `invalid_audio`, `input_audio_buffer_full`, `input_audio_buffer_commit_empty`, `too_many_pending_items`, `server_busy`, `idle_timeout`, `session_expired`.
- **Không gửi `null`** cho field optional mà type JS không cho null: bỏ hẳn `noise_reduction` và `content[].transcript`.

Dùng `conversation.item.added` (GA, docstring nêu rõ trường hợp commit buffer) thay vì `conversation.item.created` (beta). Thêm `created` nếu client thật cần.
