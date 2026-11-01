# VieGem ASR Gateway

Server nhận dạng tiếng nói tiếng Việt cho model [VieGem-ASR](https://huggingface.co/5CD-AI/VieGem-ASR) (model id `viegem-asr`), chạy bằng Docker, cung cấp một **tập con** API OpenAI:

- `GET /v1/models`, `GET /v1/models/{id}`
- `POST /v1/audio/transcriptions`: `json`, `text`, SSE (`stream=true`), `verbose_json`, `srt`, `vtt`
- `GET /v1/realtime`: Realtime transcription qua WebSocket, commit tường minh (không VAD)

Đây **không** phải bản tương thích OpenAI đầy đủ: xem [capability matrix](docs/api/README.md#capability-matrix).

> **Chỉ dùng phi thương mại.** Weights và code inference vendored (`app/viegem/`) theo [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) và [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Chạy nhanh

Cần Linux, Docker + Compose v2, NVIDIA Container Toolkit, một GPU NVIDIA (~3 GB VRAM) và [uv](https://docs.astral.sh/uv/).

```bash
uv run --extra cu126 scripts/stage_models.py                       # tải model đã pin vào ./models
mkdir -p secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/viegem_api_keys
sudo chown 10001 secrets/viegem_api_keys && sudo chmod 0400 secrets/viegem_api_keys
VIEGEM_GPU=0 docker compose up -d --build
curl -s 127.0.0.1:18000/ready                                      # {"status":"ready"}

curl -s 127.0.0.1:18000/v1/audio/transcriptions \
    -H "Authorization: Bearer $(sudo head -1 secrets/viegem_api_keys)" \
    -F model=viegem-asr -F file=@tests/fixtures/vi_a.wav
```

GPU Blackwell (RTX 50xx): dùng `--extra cu128` và `VIEGEM_BUILD_CUDA=cu128`. Chi tiết: [docs/deployment.md](docs/deployment.md).

## Giới hạn chính

- Output là text gốc của model: chữ thường, không dấu câu, số viết bằng chữ. Chỉ `language=vi`, không có logprobs.
- `srt`/`vtt` lấy thời gian từ đường CTC (start của cue có thể trễ ~1 âm tiết). `verbose_json` không có `segments`; word timestamps chưa bật.
- Realtime: transcript chỉ có sau `input_audio_buffer.commit`; không có partial, không có VAD.
- Language model KenLM (`VIEGEM_LM`) **tắt mặc định** cho tới khi qua gate.
- Gateway không làm TLS, CORS, rate limit theo client: đặt sau reverse proxy của bạn.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [deployment.md](docs/deployment.md) | Yêu cầu, CUDA profile, triển khai Docker, TLS, nhiều GPU |
| [usage.md](docs/usage.md) | Ví dụ gọi API: curl, OpenAI SDK, Realtime |
| [configuration.md](docs/configuration.md) | Biến môi trường `VIEGEM_*` |
| [operations.md](docs/operations.md) | Health/ready, dừng/cập nhật, rollback, key, log/metrics |
| [architecture.md](docs/architecture.md) | Kiến trúc và quy tắc thiết kế |
| [api/](docs/api/README.md) | API contract: [transcriptions](docs/api/transcriptions.md), [realtime](docs/api/realtime.md), [errors](docs/api/errors.md) |
| [baseline.md](docs/baseline.md) | Pins, số đo model, giới hạn frontend |
| [testing.md](docs/testing.md) | Kiểm thử và gate Docker image |
| [roadmap.md](docs/roadmap.md) | Trạng thái release và việc còn lại |
| [docs/adr/](docs/adr/) | Quyết định kiến trúc |
