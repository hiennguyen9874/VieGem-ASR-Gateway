# QUAN-Transcribe gateway (R1)

Server nhận dạng tiếng nói tiếng Việt cho [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0)
với một **tập con** API OpenAI: `GET /v1/models`, `POST /v1/audio/transcriptions` (`json`, `text`, SSE). Không
phải tương thích OpenAI đầy đủ: xem capability matrix và payload ở [docs/api-contract.vi.md](docs/api-contract.vi.md).

- Một process = một model instance = một GPU; mọi inference chạy tuần tự trên một worker thread, có queue công bằng.
- Output là text gốc của model: chữ thường, không dấu câu, số viết bằng chữ. Không có timestamps, logprobs,
  `verbose_json`/`srt`/`vtt` (R2), Realtime (R3a), chỉ `language=vi`.
- Alias `whisper-1` (tắt mặc định, `QUAN_WHISPER_ALIAS=true`) chỉ để client cũ chạy được: vẫn là model QUAN, output
  khác Whisper.

Thiết kế: [docs/deployment-design.vi.md](docs/deployment-design.vi.md), [docs/m3-design.vi.md](docs/m3-design.vi.md).
Baseline và số đo: [docs/baseline.vi.md](docs/baseline.vi.md), `docs/baseline/server.json`.

## 1. Chuẩn bị model (một lần, cần mạng)

```bash
uv sync
HF_TOKEN=<token đọc được qnguyen3/QUAN-Transcribe-1.0> uv run scripts/stage_models.py   # -> ./models, kiểm tra SHA256
uv run scripts/stage_models.py --verify                                                  # kiểm tra lại, không cần mạng
```

Revision và hash của mọi file nằm trong `assets/manifest.json`. `HF_TOKEN` chỉ dùng lúc staging; server không cần
token và không truy cập Hub.

## 2. Chạy server (local)

```bash
nvidia-smi                                   # chọn GPU còn trống
export QUAN_API_KEYS=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline uvicorn --factory app.main:create_app \
    --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --timeout-graceful-shutdown 30
curl -s localhost:8000/ready                 # {"status":"ready"} sau khi load + warm-up (~15 s)
```

Luôn `--workers 1`: muốn dùng nhiều GPU thì chạy nhiều process (mỗi process một `CUDA_VISIBLE_DEVICES`, một port)
sau load balancer. Cần `ffmpeg` trên `PATH`.

## 3. Gọi API

```bash
curl -s localhost:8000/v1/models -H "Authorization: Bearer $QUAN_API_KEYS"
curl -s localhost:8000/v1/audio/transcriptions -H "Authorization: Bearer $QUAN_API_KEYS" \
    -F model=quan-transcribe-1.0 -F file=@tests/fixtures/vi_a.wav
curl -s localhost:8000/v1/audio/transcriptions -H "Authorization: Bearer $QUAN_API_KEYS" \
    -F model=quan-transcribe-1.0 -F response_format=text -F file=@tests/fixtures/vi_a.wav
curl -sN localhost:8000/v1/audio/transcriptions -H "Authorization: Bearer $QUAN_API_KEYS" \
    -F model=quan-transcribe-1.0 -F stream=true -F file=@tests/fixtures/vi_a.wav
```

OpenAI Python SDK (`openai==3.31.0` đã kiểm thử; JS `openai@7.33.0` tương tự):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<QUAN key>")
with open("tests/fixtures/vi_a.wav", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="quan-transcribe-1.0").text)
with open("tests/fixtures/vi_a.wav", "rb") as f:
    for event in client.audio.transcriptions.create(file=f, model="quan-transcribe-1.0", stream=True):
        print(event.type, getattr(event, "delta", None) or getattr(event, "text", ""))
```

Định dạng audio: WAV, FLAC, MP3, MP4/M4A (AAC/ALAC), OGG (Vorbis/Opus/FLAC), WEBM/Matroska; nhận diện theo nội dung.
Audio dài được cắt thành chunk ≤ 30 s tại điểm yên lặng; SSE gửi một delta cho mỗi chunk xong. Audio ≤ 100 ms → `""`.
Lỗi dùng envelope OpenAI (`error.code`), danh sách ở contract §7.

## 4. Cấu hình và giới hạn

Biến `QUAN_*` được kiểm tra lúc khởi động (biến lạ, giá trị ≤ 0 hoặc không nhất quán → không start). Đầy đủ:
`app/config.py`. Những biến hay chỉnh:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `QUAN_API_KEYS` / `QUAN_API_KEYS_FILE` | (bắt buộc) | Key Bearer, ≥ 16 ký tự; file: một key mỗi dòng. Không được trùng `HF_TOKEN` |
| `QUAN_MODELS_DIR`, `QUAN_DEVICE` | `models`, `cuda` | Thư mục staging, device |
| `QUAN_MAX_UPLOAD_BYTES` | 25 MiB | Cả body multipart → 413 |
| `QUAN_MAX_AUDIO_SECONDS` | 1800 | Thời lượng sau decode → 400 `audio_too_long` |
| `QUAN_MAX_DECODED_SECONDS` | 3600 | Audio đã decode giữ trong RAM của mọi request (≈ 64 KB/s) → 429 |
| `QUAN_MAX_SPOOL_BYTES`, `QUAN_SPOOL_DIR` | 200 MiB, `/tmp` | File tạm của upload đang nhận → 429 |
| `QUAN_MAX_CHUNK_SECONDS` | 30 | Độ dài chunk (giữ 30 để giống baseline) |
| `QUAN_QUEUE_MAX_LANES` | 32 | Request đồng thời (đang decode/chờ/infer) → 429 |
| `QUAN_QUEUE_TIMEOUT_SECONDS` | 30 | Chờ chỗ trong queue/slot decode tối đa → 429 |
| `QUAN_REQUEST_TIMEOUT_SECONDS` | 600 | Tổng upload + decode + queue + infer → 503 `request_timeout` |
| `QUAN_SEND_TIMEOUT_SECONDS` | 10 | Một lần gửi SSE; client không đọc → ngắt |
| `QUAN_INFER_HANG_SECONDS` | 120 | Một call inference lâu hơn → worker "hung", `/health` 503 |
| `QUAN_SHUTDOWN_DRAIN_SECONDS` | 30 | Chờ job đang chạy khi tắt |

Các giá trị mặc định chưa được benchmark (M7); đo trên phần cứng của bạn trước khi tăng.

## 5. Docker

Image: Ubuntu 24.04 (pin digest), Python 3.12.14, FFmpeg 6.1.1, chạy user `10001`, chỉ chứa `app/`, venv và
`assets/manifest.json`. Model mount read-only lúc chạy; không có token/key trong image. Cần NVIDIA Container Toolkit.

```bash
uv run scripts/stage_models.py                              # trên host: ./models
mkdir -p secrets && python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/quan_api_keys
sudo chown 10001 secrets/quan_api_keys && sudo chmod 0400 secrets/quan_api_keys   # đọc được bởi user trong container
QUAN_GPU=<gpu> docker compose up -d --build
OPENAI_API_KEY=$(sudo cat secrets/quan_api_keys) uv run scripts/smoke_server.py --base-url http://127.0.0.1:8000
```

`compose.yaml`: rootfs read-only + tmpfs `/tmp`, `HF_HUB_OFFLINE=1`, port chỉ bind `127.0.0.1`, `stop_grace_period:
70s`. Gate container (`tests/container/test_image.py`, chạy với `QUAN_GPU=<gpu> uv run pytest tests/container`):
build, non-root, không có secret/file thừa, `--network none` → `/ready`, smoke, SIGTERM khi đang stream, restart.
**Trạng thái: chưa chạy** — host phát triển không có Docker; R1 container gate chưa đạt cho tới khi chạy được trên
host có Docker + NVIDIA Container Toolkit.

## 6. Reverse proxy và TLS

Gateway không làm TLS, rate limit theo client hay CORS. Đặt sau reverse proxy, ví dụ nginx:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_buffering off;            # SSE: không gom event (server cũng gửi X-Accel-Buffering: no)
    proxy_request_buffering off;    # stream upload tới gateway, giới hạn được kiểm tra ở gateway
    proxy_read_timeout 620s;        # >= QUAN_REQUEST_TIMEOUT_SECONDS: không có heartbeat trong lúc chờ inference
    proxy_send_timeout 620s;
    client_max_body_size 26m;       # >= QUAN_MAX_UPLOAD_BYTES (+ overhead multipart nhỏ)
}
```

Không public `/metrics` (cần API key nhưng vẫn nên chỉ cho mạng nội bộ). `/health`, `/ready` không cần key.

## 7. Vận hành và khôi phục

- `/ready` 200 khi model đã load + warm-up và không đang drain/recover: dùng cho readiness/load balancer.
- `/health` 503 `unhealthy` khi load/recovery lỗi hoặc một call inference treo quá `QUAN_INFER_HANG_SECONDS`.
  Process **không tự thoát** trong trường hợp này và Docker không restart container chỉ vì `unhealthy`: dùng
  supervisor có liveness (Kubernetes `livenessProbe` trên `/health`, systemd watchdog script, hoặc autoheal) để
  restart. Lỗi inference đơn lẻ trả 500 `inference_failed` và worker tự warm-up lại.
- Tắt (SIGTERM): ngừng nhận kết nối mới, request đang chạy được làm xong (tối đa 30 s), rồi drain queue (30 s) và
  dừng worker; log `server.stopped` có `drained`.
- Log JSON ra stderr, chỉ metadata (`request_id`, thời lượng, outcome); không ghi audio, transcript, key, tên file.
  `/metrics` (Prometheus): `quan_transcriptions_total{response_format,stream,outcome}`, `quan_jobs_total`,
  `quan_queue_wait_seconds`, `quan_infer_seconds`, `quan_sse_first_event_seconds`, `quan_ready`, `quan_worker_hung`, ….
- Xoay vòng key: thêm key mới vào file, restart, đổi client, bỏ key cũ, restart.

## 8. Kiểm thử

```bash
uv run pytest                                     # CPU: unit, scheduler, audio, contract (GPU/container tests skip)
(cd tests/contract/js && npm ci && npm test)      # JS SDK contract (server tests chạy qua pytest)
CUDA_VISIBLE_DEVICES=<gpu> uv run pytest tests/gpu
```
