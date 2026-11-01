# Vận hành

## Health và readiness

| Endpoint | Auth | Ý nghĩa |
|---|---|---|
| `GET /ready` | Không | 200 khi model (và pool LM nếu bật) đã load + warm-up, không đang drain/recover. Dùng cho load balancer |
| `GET /health` | Không | 503 khi load/recovery lỗi, inference treo quá `VIEGEM_INFER_HANG_SECONDS`, hoặc pool LM hỏng |
| `GET /metrics` | Có | Prometheus; không expose public |

Docker **không** tự restart container chỉ vì `unhealthy`: dùng watchdog/orchestrator (autoheal, systemd, Kubernetes `livenessProbe`). Lỗi inference đơn lẻ trả 500 `inference_failed` và worker tự warm-up lại.

## Dừng và cập nhật

`docker compose stop` (SIGTERM):

1. Ngừng nhận kết nối mới; request đang chạy được làm xong (≤ 30 s).
2. Drain queue (≤ 30 s); sau đó kill process LM còn lại (≤ 5 s).
3. Log `server.stopped` có `drained`.

WebSocket **không** được drain: đóng với code 1012, item chưa `completed` bị mất; client phải kết nối lại và gửi lại audio. Rolling update: bỏ instance khỏi load balancer trước.

## Rollback

Giữ image digest + CUDA profile của bản tốt. Chạy lại digest đó với cùng `./models`, chờ `/ready`, chạy smoke rồi mới mở traffic.

## Xoay vòng API key

Key chỉ đọc lúc khởi động: thêm key mới vào file → restart → chuyển client → bỏ key cũ → restart.

## Log và metrics

- Logger `viegem` ghi JSON ra stderr (`docker compose logs`), chỉ metadata (`request_id`, thời lượng, outcome). Không ghi audio, transcript, key, tên file.
- Metrics chính: `viegem_transcriptions_total`, `viegem_queue_wait_seconds`, `viegem_infer_seconds`, `viegem_ready`, `viegem_worker_hung`, `viegem_realtime_sessions`, `viegem_realtime_commit_to_completed_seconds`.
- `HF_TOKEN` chỉ dùng lúc staging, không đưa vào container.
