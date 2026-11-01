# Kiến trúc

## Lựa chọn

Inference Python của VieGem-ASR (vendored trong `app/viegem/`, revision đã pin) + một gateway FastAPI mỏng, đóng gói Docker.

- Không port sang GGUF/C++, không vLLM/Triton/Redis/Celery.
- REST, SSE và Realtime dùng chung một engine, mỗi giao thức có adapter riêng.
- Một container = một process = một model = một GPU.
- Phạm vi: ASR tiếng Việt. Dịch, TTS, chat, Realtime hội thoại không thuộc gateway này.

## Sơ đồ

```text
OpenAI SDK / curl / WebSocket client
                 |
     (HTTPS do hạ tầng của người triển khai)
                 |
   Docker container: Uvicorn, 1 worker, port 18000
   ├─ auth Bearer, request id, OpenAI error envelope    app/main.py, app/errors.py
   ├─ REST / SSE / verbose_json / srt / vtt             app/transcription.py, app/formats.py
   ├─ Realtime (explicit commit)                        app/realtime.py
   ├─ audio ingress: giới hạn body, spool, FFmpeg       app/audio.py
   ├─ chunking tại điểm yên lặng (≤ 30 s)               app/chunking.py
   ├─ scheduler: queue có giới hạn, công bằng           app/scheduler.py
   ├─ EngineWorker: thread duy nhất gọi model           app/engine.py
   └─ LMPool (tùy chọn): beam search trên process CPU   app/lm.py
                 |
   VieGem-ASR (CTC) + base google/embeddinggemma-2      app/viegem_loader.py, app/viegem/
   load từ /models (read-only, pin SHA256), không truy cập Hub
```

## Quy tắc cốt lõi

- **Mọi inference được serialize.** Model có state mutable trong forward hooks, nên chỉ `EngineWorker` (một thread) gọi model. Mỗi request/session là một *lane* trong scheduler, mỗi chunk là một *job*.
- **Event loop chỉ làm I/O.** FFmpeg chạy subprocess, inference chạy trên worker thread.
- **Công bằng:** round-robin giữa lane; lane Realtime được ưu tiên có hạn mức. Mỗi request giữ tối đa 2 chunk trong queue.
- **Mọi tài nguyên có giới hạn:** body, spool, audio trong RAM, số lane/session/job, thời lượng, deadline. Vượt → 413/429/503.
- **Hủy:** client ngắt → hủy chunk đang chờ, bỏ kết quả chunk đang chạy. Kernel đang chạy không bị ngắt; budget chỉ giải phóng khi call thật kết thúc.
- **Treo/lỗi:** call lâu hơn `VIEGEM_INFER_HANG_SECONDS` → `/health` 503. Lỗi đơn lẻ → 500, worker tự warm-up lại.

## Luồng xử lý

**REST/SSE**

1. Auth, giới hạn body (trước parse multipart), validate tham số theo thứ tự cố định.
2. Spool upload; FFmpeg (argv cố định, chỉ protocol `file`, allowlist demuxer/decoder) decode về mono 16 kHz float32.
3. `chunking.split`: chunk ≤ 30 s, cắt tại cửa sổ 50 ms yên lặng nhất trong 5 s cuối; các chunk phủ liền audio. Audio ≤ 100 ms → `""`.
4. `run_chunks` đưa từng chunk qua lane; `Joiner` nối text bằng một dấu cách.
5. SSE: headers gửi sau khi decode xong; một `delta` mỗi chunk, một `done` ở cuối.

**Timestamps (`srt`/`vtt`)**: từ đường CTC của chính lần inference sinh ra text (`app/alignment.py`), cộng offset của chunk; không chạy thêm GPU. Frame 40 ms. Đường không khớp transcript → 500 `alignment_failed`, không bao giờ chia đều thời gian.

**Realtime**: kiểm tra auth/model/capacity **trước** accept. Mỗi session giữ một lane LIVE. PCM16 24 kHz được resample stream sang 16 kHz (soxr); commit tạo item bất biến, xử lý FIFO qua cùng `split`/`run_chunks`/`Joiner` như REST.

**LM (`VIEGEM_LM=true`, ADR 0002)**: engine thread chỉ chạy model và trả log-prob đã prune; scheduler giải phóng GPU rồi đưa cho `LMPool` chạy prefix beam search + forced alignment. Kết quả cùng kiểu `ChunkResult`, nên mọi format và Realtime không đổi.

## Model và dữ liệu

- Load bằng `app/viegem_loader.load_viegem` từ thư mục đã stage (`weights_only`, không resolve Hub id). Parity với code reference đã kiểm chứng (`tests/gpu/test_viegem_parity.py`). Lý do vendor code: ADR 0001.
- Nguồn pin duy nhất: `assets/manifest.json`; `scripts/stage_models.py` tải và kiểm tra. Xem [baseline.md](baseline.md).
- Peak đo được: ~2.4 GB VRAM, ~2.5 GB RSS lúc load.

## Đóng gói

- Image multi-stage: builder `uv sync --frozen --no-dev --extra <profile>`; runtime Ubuntu 24.04 (pin digest) + FFmpeg 6.1.1, chỉ copy `app/` và venv.
- Non-root UID 10001, rootfs read-only, tmpfs `/tmp`, model read-only, key qua Docker secret, `HF_HUB_OFFLINE=1`; chạy được với `--network none`.
- `HEALTHCHECK` trên `/health`; `stop_grace_period: 70s` = Uvicorn 30 s + drain 30 s + dừng worker 5 s.
- Uvicorn luôn `--workers 1 --ws websockets-sansio --ws-max-size 2097152` (`auto` có thể âm thầm tắt WebSocket).

## Bảo mật

- Không có API key mặc định; key ≥ 16 ký tự, không được trùng `HF_TOKEN`.
- Không log audio, transcript, key, tên file; tên file client không dùng làm path.
- Realtime chỉ server-to-server (Bearer header), không có ephemeral client secret.
- TLS, CORS, rate limit theo client thuộc hạ tầng phía trước.
