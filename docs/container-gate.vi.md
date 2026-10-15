# M3 — Kết quả container gate

## Trạng thái (2026-10-15)

**R1 GPU container gate đã đạt với profile `cu128` trên RTX 5090.** Giữ model/code pins và baseline cu126,
thêm profile riêng torch 2.11.0+cu128 vì PyTorch không phát hành wheel 2.14.1 cho CUDA 12.8. Không sửa host driver.

## Hai profile hiện tại

| Profile | PyTorch | Image SHA256 | Kết quả |
|---|---|---|---|
| cu126 | 2.14.1+cu126 | `605c318bf3970599fce848cf3ecd29a892edd57f5c604af717c23c8d8df87489` | **6 CPU tests passed, 114.75 s**; không hỗ trợ RTX 5090 |
| cu128 | 2.11.0+cu128 | `bcda928e01e1ab2ea661412799f70d59d9a8d5492eea8b4b9a7274564424b3d2` | **6 GPU container tests passed, 28.30 s, không skip** |

`uv sync/run --extra cu126` hoặc `--extra cu128` (không dùng cả hai, không bỏ extra).
Docker mặc định cu126; chọn qua `--build-arg TORCH_CUDA=cu128`. Compose chọn `QUAN_BUILD_CUDA=cu128`;
không chuyển biến build/test vào env server. Mọi package lấy từ `uv.lock`, indexes explicit, hai extras xung đột.
Build arg ngoài hai giá trị bị từ chối; `uv sync --extra cu126 --extra cu128` cũng bị từ chối.
`docker compose config --quiet` pass cho cả hai profiles.

```bash
QUAN_GPU=0 uv run --extra cu128 --offline pytest tests/container --container-cuda=cu128 -v
uv run --extra cu126 --offline pytest tests/container --container-device=cpu -v
```

GPU gate dùng network none, read-only rootfs/models, secret file; smoke 5 fixtures (REST formats/SSE/Realtime)
khớp baseline, auth/errors/overload/isolation, SIGTERM trong stream 6 × vi_long.wav → SSE done, exit 0,
`drained: true`, restart → ready. `sm_120` có trong `torch.cuda.get_arch_list()` của cu128.
Bằng chứng chỉ metadata (không transcripts): [baseline/server-container-cu128.json](baseline/server-container-cu128.json).
Report phân tách profile/device: cu126 GPU dùng `server-container.json`, cu128 GPU dùng `server-container-cu128.json`,
CPU thêm `-cpu` trước suffix profile; CPU không bao giờ ghi đè GPU evidence.

## Host và lần chạy ban đầu (cu126)

- Docker Engine 28.4.0, Compose 2.39.4; NVIDIA Container Toolkit/runtime có sẵn.
- Host: RTX 5090 (`sm_120`), driver 575.64.03 (CUDA driver API 12.9). Chọn GPU 0 còn khoảng 16 GiB trống;
  không dừng hay sửa các service khác.
- Image `quan-transcribe:test`: `sha256:cf93692be5af415cb118ca61cc7b93900b02607fd08d70dd640b93d84f4772a0`
  (~7.57 GB), Python 3.12.14, torch 2.14.1+cu126, FFmpeg 6.1.1.
- `stage_models.py` và `fetch_fixtures.py`: toàn bộ model/fixture (gồm `vi_long.wav`) khớp SHA256 manifest.

## Blocker lịch sử: GPU với cu126

```bash
QUAN_GPU=0 uv run --extra cu126 --offline pytest tests/container -v
```

**3 passed, 3 failed**: non-root UID 10001, minimal copy và không có secrets trong image đều đạt.
Cả ba runtime tests dừng ở warm-up: `engine.load_failed`, `AcceleratorError`,
`CUDA error: no kernel image is available for execution on the device`; `/ready` và `/health` trả 503 đúng policy.

`torch.cuda.get_arch_list()` của wheel pin chỉ gồm `sm_50` … `sm_90`, không có `sm_120`.
Index PyTorch không có wheel 2.14.1 cho cu128/cu129; cu130 có wheel cp312 Linux x86_64 nhưng cần driver mới hơn.
Lần đầu giữ nguyên torch, không đổi sang cu130 trên driver 575, không coi CPU fallback là GPU gate pass.
Profile cu128 mới giải quyết blocker bằng torch 2.11.0 riêng; cu126 vẫn giữ nguyên 2.14.1.

## CPU diagnostics (không thay thế GPU gate)

```bash
uv run --extra cu126 --offline pytest tests/container --container-device=cpu -v
```

**6 passed trong 114.75 s**, gồm SSE `done` khi SIGTERM, exit code 0, `drained: true`, restart → ready.

Chế độ CPU opt-in không cấp GPU cho container, giới hạn OMP/MKL 2 threads, vẫn giữ network none, rootfs read-only,
models mount read-only, tmpfs và key qua file secret (như compose). Smoke kiểm tra JSON/text/SSE/verbose_json/SRT/VTT
và Realtime trên cả 5 fixtures. Test riêng kiểm tra auth 401, model 404, invalid audio 400, overload 429 bằng hai
realtime lanes chiếm chỗ, rồi hai REST request đồng thời khớp baseline riêng (không trộn kết quả).

Drain workload GPU giữ nguyên 6 × `vi_long.wav`; CPU diagnostics dùng 2 × để vừa ngân sách Uvicorn 30 s.
Lần thử CPU với 6 × không phát `done` trước khi shutdown timeout, nên không được tính pass.
Report CPU riêng: [`baseline/server-container-cpu.json`](baseline/server-container-cpu.json), chỉ metadata/timings/counts,
không lưu audio/transcript; trường `gpu_release_gate: false`. Report GPU `server-container.json` chỉ được tạo sau
GPU smoke thật thành công.

## Regression

Unit/scheduler/audio/contract sau khi thêm hai profiles: **407 passed trong 34.56 s, không skip**, với Python 3.12.14
trích từ image, dùng packages cu128 từ venv host; host có FFmpeg/OpenSSL và Node 22.23.3.
Đã `npm ci` + `npm test` (tsc strict và mocked examples); suite pytest gồm cả JS SDK integration với server HTTP/WSS thật.
Không cần CUDA.
Python mặc định host là 3.12.3: lần chạy trên bản này có 1 failure validation base64 trong Realtime; chạy lại trên
3.12.14 đúng pin baseline thì test đó và toàn bộ suite pass. Image production không cài dev/test tools.

## Giới hạn và chạy lại

- cu126 vẫn không hỗ trợ RTX 5090. GPU gate cu126 mới cần GPU tương thích (ví dụ A100 của baseline);
  CPU diagnostics không phải bằng chứng GPU cu126. cu128 GPU pass không có nghĩa cu126 chạy trên Blackwell.
- Khi thay torch/CUDA/profile pins, chạy lại unit/contract và toàn bộ container gate với profile đích **không skip**;
  so transcript trong RAM với baseline, không lưu/log transcripts. Giữ model/code pins độc lập với lựa chọn CUDA.
- Gate kiểm tra Docker run; compose config đã validate, chưa triển khai compose service thực trên host dùng chung.

`docker compose config --quiet` đã pass; chưa chạy compose service trên host (port 18000/service khác không bị đụng tới).
Các runtime tests không publish port, client sidecar dùng chung network namespace nên vẫn chạy khi gateway `--network none`.
