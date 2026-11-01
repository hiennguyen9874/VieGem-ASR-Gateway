# Kiểm thử

Lệnh phát triển đầy đủ (setup, server local, probe): `docs/agents/commands.md`.

## Các tầng test

```bash
uv run --extra cu126 pytest                                  # CPU: unit, scheduler, audio, contract (cần ffmpeg)
(cd tests/contract/js && npm ci && npm test)                 # JS SDK contract
CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 pytest tests/gpu
VIEGEM_GPU=<gpu> uv run --extra cu128 --offline pytest tests/container --container-cuda=cu128   # image gate
```

| Thư mục | Nội dung |
|---|---|
| `tests/unit`, `tests/scheduler`, `tests/audio` | Logic thuần, scheduler, FFmpeg/chunking (fake engine) |
| `tests/contract` | Payload vs Python/JS SDK thật ([api/](api/README.md#kiểm-tra-contract)) |
| `tests/gpu` | Parity model, engine, server, Realtime, LM trên GPU |
| `tests/container` | Gate Docker image |

## Gate Docker image

Phải đạt trước khi phát hành image: `tests/container/test_image.py`.

**Cần:** Docker, NVIDIA Container Toolkit, GPU trống, model đã stage (`./models`), fixtures (`scripts/fetch_fixtures.py`, gồm `fixtures-large/vi_long.wav`).

```bash
VIEGEM_GPU=<gpu> uv run --extra cu128 --offline pytest tests/container --container-cuda=cu128 -v
VIEGEM_GPU=<gpu> uv run --extra cu126 --offline pytest tests/container -v              # GPU sm_≤90
uv run --extra cu126 --offline pytest tests/container --container-device=cpu -v     # chẩn đoán, không phải gate
```

Report GPU ghi vào `docs/baseline/server-container[-cu128].json` (chỉ metadata). Report CPU không commit.

| Test | Kiểm tra |
|---|---|
| Image | Build đúng profile; UID 10001; `/srv` chỉ có `app/` và `.venv`; không có references, models, tests, `.git`, secrets; đúng phiên bản Python, FFmpeg, torch/CUDA |
| Offline + smoke | `--network none`, rootfs/models read-only, key qua secret → `/ready`; `smoke_server.py` trên 5 fixture: mọi format + Realtime khớp baseline |
| Auth / lỗi / overload / isolation | 401, 404 model, 400 audio lỗi, 429 khi hết lane; hai request đồng thời không trộn transcript |
| Shutdown + restart | SIGTERM khi đang stream 6 × `vi_long.wav` → SSE `done`, exit 0, `drained: true`; restart → ready |

### Trạng thái

| Profile | Kết quả |
|---|---|
| `cu128` | Đạt trên RTX 5090 (6 passed) — nhưng là image **trước khi đổi tên** gateway (QUAN) |
| `cu126` | Chưa chạy GPU gate (cần GPU ≤ `sm_90`, vd. A100); chỉ chạy CPU diagnostic |

Cần chạy lại gate cho VieGem-ASR trước khi phát hành (#5). Khi đổi torch/CUDA/base image/FFmpeg: chạy lại unit/contract và toàn bộ gate với profile đích, không skip.
