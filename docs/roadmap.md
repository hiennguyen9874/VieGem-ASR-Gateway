# Roadmap

Release hiện tại: `v0.1.0-rc.1` (pre-release). Một tính năng chỉ được bật sau khi gate của nó đạt và có bằng chứng ghi lại.

## Trạng thái

| Release | Nội dung | Trạng thái |
|---|---|---|
| R1 | REST `json`/`text`, SSE, auth, lỗi OpenAI, giới hạn, Docker một GPU | **Xong** |
| R2a | `verbose_json` (không `segments`), `srt`, `vtt` với timestamps CTC | **Xong** |
| R3a | Realtime WebSocket, explicit commit | **Xong** |
| R2b | Word timestamps | Chờ bộ annotate tay |
| R3b | Server VAD, partial | Chưa làm |
| R4 | Capacity đo trên GPU riêng | Có tooling, chưa đo |

## Bằng chứng hiện có

- **CPU suite:** unit + scheduler + audio + contract (SDK Python/JS thật, server HTTP/WSS thật) pass.
- **GPU tier (A100, fp32):** 18 passed, gồm parity, silence, Realtime, LM. Server smoke và alignment probe 0 failure.
- **WER fixture:** 20/329, vẫn vượt ngưỡng QUAN 17/329; ngoại lệ đã chấp nhận ở #4 ([baseline](baseline.md)).
- **Container gate:** chỉ có bằng chứng cu128 của image cũ (QUAN); cần chạy lại cu126 và cu128 (#5, [testing](testing.md#gate-docker-image)).
- **LM (#8):** code có, tắt mặc định; chưa qua gate.

## Lệch so với thiết kế ban đầu

- `verbose_json` không có `segments`: schema SDK bắt buộc field kiểu Whisper mà model không có dữ liệu thật.
- Realtime phát `conversation.item.added` (GA) thay vì `conversation.item.created` (beta).
- Không dùng `LiveSession` của reference: Realtime chỉ infer khi commit, qua cùng splitter/joiner với REST.
- Thêm profile `cu128` (torch 2.11.0) cho Blackwell vì không có wheel 2.14.1 cho CUDA 12.8.

## Việc còn lại (theo ưu tiên)

### 1. Capacity (R4)

Mặc định hiện tại (`VIEGEM_QUEUE_MAX_LANES=32`, `VIEGEM_MAX_SESSIONS=16`, …) là giới hạn an toàn, **chưa đo**.

- [ ] Chạy GPU container gate cu126 trên A100/H100 và cu128 cho image mới.
- [ ] Benchmark trên GPU **riêng**: REST ngắn/dài, SSE, WebSocket, mixed; ≥ 3 lần mỗi mức, ≥ 100 thành công mỗi nhóm cho p95. Chốt workload, SLO p95, error budget trước khi đo.
- [ ] Soak ≥ 1 h sau TLS thật; theo dõi RSS/VRAM.
- [ ] Diễn tập lỗi: SIGTERM drain, kill/restart, CUDA OOM, watchdog worker treo.
- [ ] Đặt giới hạn từ số đo; ghi bảng quyết định (phần cứng, image digest, config, tải tối đa, giới hạn chọn) và digest rollback.

```bash
uv run --extra cu128 scripts/benchmark.py --engine gpu --hardware '<gpu>-exclusive-cu128' \
  --base-url http://127.0.0.1:18000 --clients 1,2,4,8,16,32 --requests-per-client 100 \
  --workload mixed --phase warm --server-pid <host pid> --out docs/baseline/capacity-mixed-warm.json
```

Report benchmark (exit 0) không phải tuyên bố capacity; số client là closed-loop.

**Chạy chẩn đoán (không phải gate):** A100 dùng chung, model QUAN, 30 request/client, 0 lỗi.

| Clients | Ngắn 8.88 s: audio s/s · p50 · p95 | Dài 106.62 s: audio s/s · p50 · p95 |
|---:|---|---|
| 1 | 71 · 0.12 · 0.13 s | 279 · 0.39 · 0.40 s |
| 2 | 128 · 0.13 · 0.16 s | 412 · 0.50 · 0.62 s |
| 4 | 137 · 0.25 · 0.28 s | 421 · 1.00 · 1.14 s |
| 8 | 132 · 0.52 · 0.62 s | 426 · 1.99 · 2.08 s |
| 16 | 136 · 1.03 · 1.16 s | 413 · 4.10 · 4.36 s |

Throughput bão hòa ở ~2 client; sau đó latency tăng tuyến tính. GPU chỉ ~38% utilization: thread model bị giới hạn bởi kernel launch (~50 ms mỗi call), nên các tối ưu ở mục 4 nâng trần này.

### 2. Word timestamps (R2b)

`timestamp_granularities[]=word` trả 400 cho tới khi gate đạt (`WORD_TIMESTAMPS = False` trong `app/transcription.py`). Code alignment và evaluator (`scripts/eval_alignment.py`) đã có.

- [ ] Annotate tay ranh giới từ trong `tests/fixtures/alignment/` (Praat/Audacity, không dùng aligner khác): ≥ 200 từ, ≥ 10 utterance, chia `dev`/`test`.
- [ ] Gate trên `test`: ≥ 90% từ lệch ≤ 100 ms, median ≤ 50 ms (ngưỡng chốt trước).
- [ ] Bật switch trong commit chứa report; thêm `words` vào `verbose_json`.

### 3. Server VAD và partial (R3b)

- [ ] VAD âm học (pin model, kiểm license và chi phí CPU) với `threshold`, `prefix_padding_ms`, `silence_duration_ms`; phát `speech_started/stopped` và auto-commit. Không dùng CTC blank làm VAD. `semantic_vad` vẫn từ chối.
- [ ] Tùy chọn: partial trước commit bằng decoder stable-prefix, có chính sách sửa final rõ ràng. Ứng viên: streaming mode của VieGem-ASR (FLEURS WER 9.02 so với 7.33 full context), cần gate riêng.

### 4. Scale và hiệu năng

- [ ] Nhiều GPU: mỗi GPU một container sau load balancer, sticky WebSocket.
- [ ] Micro-batching chunk từ nhiều request (8 chunk/call chỉ 60.5 ms so với ~50 ms cho 1 chunk); cần parity và deadline theo request.
- [ ] Giảm overhead launch: CUDA graphs hoặc `torch.compile` với shape cố định; gate bằng text giống hệt `offline.json`.
- [ ] Gate LM: đo WER fixture, p95 decode mỗi chunk 30 s, RTF; chốt ngưỡng rồi quyết định mặc định.

### 5. Việc nhỏ

- [ ] `conversation.item.created` nếu client thật cần.
- [ ] Input G.711 `audio/pcmu` / `audio/pcma`.
- [ ] Truy cập từ trình duyệt: ephemeral client secret hoặc token service.
- [ ] Hậu xử lý dấu câu / inverse text normalization (riêng, có version).
- [ ] Hỗ trợ Python 3.11.

## Ngoài phạm vi

Model không cung cấp; cần engine khác:

- Ngôn ngữ khác tiếng Việt, auto-detect, `/v1/audio/translations`.
- `/v1/audio/speech`, chat/responses, Realtime hội thoại, WebRTC/SIP.
- Diarization, `prompt`/keywords biasing, `include[]=logprobs`.
- `verbose_json.segments` (cần field chỉ Whisper có).
- TLS, CORS, rate limit theo client trong gateway.
