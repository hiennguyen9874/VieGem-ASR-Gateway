# Cấu hình

Thêm biến `VIEGEM_*` vào `environment:` của service (ví dụ trong `compose.override.yaml`). Định nghĩa đầy đủ: `app/config.py`.

Container **không start** nếu có biến lạ (kể cả `QUAN_*` cũ), giá trị ≤ 0 hoặc cấu hình không nhất quán.

## Chung

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIEGEM_API_KEYS_FILE` / `VIEGEM_API_KEYS` | (bắt buộc) | Key Bearer, ≥ 16 ký tự ASCII printable (compose dùng file secret) |
| `VIEGEM_MODELS_DIR` | `models` | Thư mục model đã stage (compose: `/models`) |
| `VIEGEM_DEVICE` | `cuda` | Thiết bị chạy model |
| `VIEGEM_PRECISION` | `fp32` | `fp32` (WER thấp nhất) hoặc `bf16` (file dài nhanh ~2.6×); `fp16` bị từ chối. Xem [baseline](baseline.md#precision) |
| `VIEGEM_WHISPER_ALIAS` | `false` | Chấp nhận model id `whisper-1` |
| `VIEGEM_LOG_LEVEL` | `INFO` | Mức log |

## Language model (chưa qua gate)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIEGEM_LM` | `false` | KenLM beam search trên pool CPU (alpha 0.35, beta 1.5, beam 32). Cần file LM đã stage, thiếu → không start |
| `VIEGEM_LM_WORKERS` | 2 | Số process CPU (≤ `VIEGEM_LM_MAX_PENDING`, ≤ số CPU) |
| `VIEGEM_LM_MAX_PENDING` | 8 | Chunk chờ + đang decode; đầy → chờ `VIEGEM_QUEUE_TIMEOUT_SECONDS` rồi 429 |

## Giới hạn request

| Biến | Mặc định | Vượt → |
|---|---|---|
| `VIEGEM_MAX_UPLOAD_BYTES` | 25 MiB | 413 (cả body multipart) |
| `VIEGEM_MAX_AUDIO_SECONDS` | 1800 | 400 `audio_too_long` |
| `VIEGEM_MAX_DECODED_SECONDS` | 3600 | 429 (audio đã decode trong RAM của mọi request, ≈ 64 KB/s) |
| `VIEGEM_MAX_SPOOL_BYTES` | 200 MiB | 429 (file tạm upload, trong tmpfs 512 MiB) |
| `VIEGEM_MAX_CHUNK_SECONDS` | 30 | Độ dài chunk (giữ 30 để giống baseline) |
| `VIEGEM_QUEUE_MAX_LANES` | 32 | 429 (request/session đồng thời) |
| `VIEGEM_QUEUE_TIMEOUT_SECONDS` | 30 | 429 (chờ chỗ trong queue) |
| `VIEGEM_REQUEST_TIMEOUT_SECONDS` | 600 | 503 `request_timeout` (upload + decode + queue + infer + gửi) |
| `VIEGEM_SEND_TIMEOUT_SECONDS` | 10 | Ngắt kết nối (một lần gửi SSE/WebSocket) |

## Realtime

| Biến | Mặc định | Vượt → |
|---|---|---|
| `VIEGEM_MAX_SESSIONS` | 16 | Handshake 429 (≤ lanes) |
| `VIEGEM_MAX_SESSION_SECONDS` | 3600 | `session_expired` |
| `VIEGEM_SESSION_IDLE_TIMEOUT_SECONDS` | 120 | `idle_timeout` |
| `VIEGEM_MAX_TURN_SECONDS` | 120 | `input_audio_buffer_full` |
| `VIEGEM_MAX_PENDING_ITEMS` | 4 | `too_many_pending_items` |

## Worker

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIEGEM_INFER_HANG_SECONDS` | 120 | Call inference lâu hơn → worker "hung", `/health` 503 |
| `VIEGEM_SHUTDOWN_DRAIN_SECONDS` | 30 | Chờ job đang chạy khi tắt |

Các mặc định là giới hạn an toàn, **chưa phải capacity đã đo** ([roadmap](roadmap.md#1-capacity-r4)). Đo trên phần cứng của bạn trước khi tăng.
