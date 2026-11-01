# Triển khai

## Yêu cầu

- Linux, Docker Engine + Compose v2, NVIDIA Container Toolkit, một GPU NVIDIA (~3 GB VRAM mỗi instance). Không cần CUDA toolkit trên host.
- [uv](https://docs.astral.sh/uv/) trên host, chỉ để tải model và chạy smoke test.
- License: chỉ phi thương mại (CC-BY-NC-SA-4.0 + Gemma Terms of Use). Repo không phân phối weights.

## Chọn CUDA profile

| Profile | PyTorch | GPU | Driver tối thiểu |
|---|---|---|---|
| `cu126` (mặc định) | 2.14.1+cu126 | tới `sm_90` (A100, H100, …) | 560.28.03 |
| `cu128` | 2.11.0+cu128 | thêm Blackwell `sm_120` (RTX 5090, …) | 570.26 |

Mọi lệnh `uv run`/`uv sync` kèm **đúng một** `--extra cu126` hoặc `--extra cu128` (không chạy trần, không `--all-extras`).

## Các bước

```bash
# 1. Tải model đã pin (gồm LM ~71 MB) vào ./models, kiểm tra SHA256. HF_TOKEN=<token> nếu cần.
uv run --extra cu126 scripts/stage_models.py
uv run --extra cu126 scripts/stage_models.py --verify    # kiểm tra lại, không cần mạng

# 2. API key: một key mỗi dòng, ≥ 16 ký tự, UID 10001 đọc được
mkdir -p secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/viegem_api_keys
sudo chown 10001 secrets/viegem_api_keys && sudo chmod 0400 secrets/viegem_api_keys

# 3. Build + chạy (chọn GPU trống bằng nvidia-smi)
VIEGEM_GPU=0 docker compose up -d --build                          # cu126
VIEGEM_BUILD_CUDA=cu128 VIEGEM_GPU=0 docker compose up -d --build  # Blackwell

# 4. Chờ sẵn sàng (load + warm-up)
curl -s 127.0.0.1:18000/ready

# 5. Smoke test (tùy chọn): mọi format + Realtime, so với baseline
OPENAI_API_KEY=$(sudo head -1 secrets/viegem_api_keys) \
    uv run --extra cu126 scripts/smoke_server.py --base-url http://127.0.0.1:18000
```

`VIEGEM_BUILD_CUDA` và `VIEGEM_GPU` chỉ dùng cho compose. Compose luôn mount `./models`; nếu stage sang thư mục khác thì sửa volume.

## Image và compose

Image `viegem-asr-gateway:r1-<profile>`: Ubuntu 24.04 (pin digest), Python 3.12.14, FFmpeg 6.1.1, user `10001`, chỉ chứa `app/` và venv. Không có model, `HF_TOKEN` hay API key trong image; server không truy cập Hub.

`compose.yaml`:

- model mount `./models:/models:ro`; key qua Docker secret `/run/secrets/viegem_api_keys`;
- rootfs read-only, tmpfs `/tmp` 512 MiB (spool upload), `HF_HUB_OFFLINE=1`;
- port chỉ bind `127.0.0.1:18000`; `restart: unless-stopped`; `stop_grace_period: 70s`.

## Scale

**Một container = một model = một GPU**, luôn một Uvicorn worker. Nhiều GPU: chạy nhiều container (mỗi container một `VIEGEM_GPU`, một port) sau load balancer; WebSocket cần sticky routing.

## TLS và reverse proxy

Khi expose ra ngoài localhost, đặt gateway sau HTTPS proxy với:

- SSE không buffer; hỗ trợ WebSocket upgrade (JS SDK Realtime chỉ kết nối `wss:`);
- read timeout ≥ `VIEGEM_REQUEST_TIMEOUT_SECONDS` (600 s);
- giới hạn body ≥ `VIEGEM_MAX_UPLOAD_BYTES` (25 MiB).
