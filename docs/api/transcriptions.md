# `POST /v1/audio/transcriptions`

Auth: `Authorization: Bearer <key>`. Body `multipart/form-data`. Lỗi: [errors.md](errors.md).

## Tham số

| Field | Quy tắc |
|---|---|
| `file` | Bắt buộc, đúng một part. Tên file bị bỏ qua (không dùng làm path, không log). Nhận diện theo nội dung. Container: WAV, FLAC, MP3/MPGA, MP4/M4A, OGG, WEBM/Matroska. Decoder: PCM (`s16le`, `s24le`, `s32le`, `u8`, `f32le`, `f64le`, A-law, µ-law), FLAC, ALAC, MP1/2/3, AAC, Vorbis, Opus. Mọi tổ hợp FFmpeg đọc được đều nhận. Ngoài allowlist (ADTS `.aac`, MPEG-PS, playlist, ADPCM, …) → 400 `invalid_audio`. Đưa về mono 16 kHz |
| `model` | Bắt buộc; thiếu → 400 `missing_required_parameter`; không phục vụ → 404 `model_not_found` |
| `language` | Không gửi hoặc `vi`; khác → 400 `unsupported_language`. Không gửi ≠ auto-detect |
| `temperature` | Không gửi hoặc bằng 0 (`0`, `0.0`, `-0`). Không phải số hữu hạn → 400 `invalid_value`; khác 0 → 400 `unsupported_parameter` |
| `prompt` | Không gửi hoặc rỗng; khác → 400 `unsupported_parameter` |
| `response_format` | `json` (mặc định), `text`, `verbose_json`, `srt`, `vtt`; khác → 400 `unsupported_response_format` |
| `stream` | Đúng `true` hoặc `false`; khác → 400 `invalid_value`. `true` với `json`/`text` → SSE; với `verbose_json`/`srt`/`vtt` → 400 `unsupported_parameter` |
| `include[]` | Field lặp, tên có `[]`. Mọi giá trị → 400 `unsupported_parameter` |
| `timestamp_granularities[]` | Field lặp, giá trị gộp. Ngoài {`word`, `segment`} → 400 `invalid_value`; format ≠ `verbose_json`, `segment`, hoặc `word` (chưa bật) → 400 `unsupported_parameter` |
| Field khác | 400 `unsupported_parameter` (`param` = tên field, cắt 64 ký tự). Không âm thầm bỏ qua |

Mọi field trừ `file`, `include[]`, `timestamp_granularities[]` chỉ gửi một lần; lặp → 400 `invalid_value`.

**Thứ tự kiểm tra** (trả lỗi đầu tiên; `app/transcription.py:parse_params`, `tests/unit/test_transcription_params.py`): multipart/`file` → (1) field lạ → (2) field lặp → (3) `model` thiếu/rỗng → (4) `include[]` → (5) giá trị `timestamp_granularities[]` → (6) `language` → (7) `temperature` → (8) `prompt` → (9) `response_format` → (10) `stream` → (11) granularity chỉ với `verbose_json` → (12) `segment`/`word` → (13) model không phục vụ (404) → decode audio.

## Kết quả

| Format | Response |
|---|---|
| `json` | `application/json`: `{"text", "usage": {"type": "duration", "seconds"}}`. `seconds` = số mẫu sau decode / sample rate gốc |
| `text` | `text/plain; charset=utf-8`, đúng transcript, không thêm newline |
| `verbose_json` | `{"text", "language": "vietnamese", "duration", "usage"}`, `duration == usage.seconds`. **Không có `segments`**, chưa có `words` |
| `srt` | `text/plain`: `n\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n`, đánh số từ 1; không có cue → body rỗng |
| `vtt` | `text/plain`: `WEBVTT\n\n` rồi `HH:MM:SS.mmm --> HH:MM:SS.mmm\ntext\n\n`; không có cue → chỉ header |

Ví dụ: `examples/transcription.json`, `transcription_verbose.json`, `transcription.srt`, `transcription.vtt` (dạng R2b có `words`, chưa phục vụ: `transcription_verbose_words.json`).

**Text:** chữ thường, không dấu câu, số viết bằng chữ. Nhiều chunk nối bằng một dấu cách; chunk rỗng không sinh dấu cách. Mọi format cho cùng transcript: `json.text == text == verbose_json.text == SSE done.text ==` text các cue nối bằng dấu cách.

## Timestamps (`srt`/`vtt`)

- Lấy từ đường CTC greedy của chính lần inference (khi `VIEGEM_LM=true`: forced alignment của LM trên cùng log-prob), cộng offset của chunk.
- **Từ** = dãy ký tự không phải khoảng trắng (âm tiết). Thời gian của từ = các frame 40 ms model phát unit của nó, nên start của cue có thể trễ ~1 âm tiết (`docs/baseline/alignment_probe.json`).
- Cue mới khi khoảng lặng giữa hai từ ≥ 0.5 s, hoặc cue sẽ dài > 7 s hay > 80 ký tự. Một từ tự nó vượt giới hạn vẫn giữ nguyên trong cue riêng.
- Giờ là ms nguyên làm tròn; cue có thứ tự, không chồng nhau, nằm trong `[0, duration]`.
- Đường CTC không khớp transcript → 500 `alignment_failed` (chỉ `srt`/`vtt`).

## Audio ngắn, im lặng, chunking

- Audio ≤ 1,600 mẫu ở 16 kHz (≤ 100 ms), kể cả 0 mẫu → `text: ""`, không gọi model (frontend crash dưới 161 mẫu).
- Chunk có mọi mẫu bằng 0 → text và đường CTC rỗng. Không dùng ngưỡng biên độ: audio khác 0 dù rất nhỏ vẫn decode và có thể cho từ sai ([baseline](../baseline.md#giới-hạn-frontend-và-im-lặng)).
- Audio dài hơn `VIEGEM_MAX_CHUNK_SECONDS` (30 s) được cắt tại điểm yên lặng nhất trong 5 s cuối; các chunk phủ liền audio, mỗi chunk > 100 ms. Text có thể khác `transcribe()` của reference vì điểm cắt khác.

## Giới hạn

| Điều kiện | Kết quả |
|---|---|
| Body > `VIEGEM_MAX_UPLOAD_BYTES` (25 MiB, kể cả chunked) | 413 |
| > 32 part, field không phải file > 8 KiB | 413; field phải là UTF-8 |
| Audio > `VIEGEM_MAX_AUDIO_SECONDS` (1800 s) | 400 `audio_too_long` |
| Decode quá `VIEGEM_DECODE_TIMEOUT_SECONDS` | 400 `invalid_audio` |
| Tổng thời gian > `VIEGEM_REQUEST_TIMEOUT_SECONDS` | 503 hoặc SSE error `request_timeout` |
| Hết slot decode, spool, lane, hoặc budget `VIEGEM_MAX_DECODED_SECONDS` | 429 `server_busy` |

Stream bị cụt được decode phần đọc được. Khi client ngắt hoặc hết hạn, budget của waveform được giữ tới khi mọi job của lane kết thúc.

## SSE (`stream=true`)

- Validate **và decode xong trước khi gửi headers**; lỗi giai đoạn này là HTTP lỗi.
- Headers: `200`, `text/event-stream; charset=utf-8`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`, chunked. Không heartbeat.
- Mỗi event là `data: <json>` + dòng trống; không có `event:`, không gửi `[DONE]`.
- `{"type": "transcript.text.delta", "delta"}`: một event mỗi chunk có text, theo thứ tự; delta thứ hai trở đi bắt đầu bằng dấu cách.
- `{"type": "transcript.text.done", "text"}`: đúng một lần ở cuối; `"".join(deltas) == text`. Không có `usage`.
- Audio rỗng/ngắn: chỉ một `done` với `text: ""`.
- Lỗi sau headers: một event `data: {"error": {...}}` rồi đóng, **không** có `done`. SDK ném `APIError`.
- Client ngắt hoặc không đọc quá `VIEGEM_SEND_TIMEOUT_SECONDS`: hủy chunk đang chờ, bỏ kết quả đang chạy, không gửi gì thêm.
- Tối đa 2 chunk của một request trong queue cùng lúc; server không buffer thêm event.

Ví dụ: `examples/transcription_stream.sse`, `examples/transcription_stream_error.sse`.
