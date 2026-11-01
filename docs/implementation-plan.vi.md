# Trạng thái triển khai VieGem ASR Gateway (model VieGem-ASR)

Kiến trúc: [deployment-design.vi.md](deployment-design.vi.md). Việc còn lại: [roadmap.md](roadmap.md).
Pre-release: tag `v0.1.0-rc.1`.

## 1. Release

| Release | Nội dung | Trạng thái |
|---|---|---|
| R1 — REST/SSE | models, transcription `json`/`text`, SSE, auth, lỗi OpenAI, giới hạn, Docker một GPU | **Xong** |
| R2a — Rich output | `verbose_json` (không `segments`), `srt`, `vtt` với timestamps từ CTC | **Xong** |
| R2b — Word timestamps | `timestamp_granularities[]=word` | Chưa: chờ bộ annotate tay |
| R3a — Realtime explicit commit | WS `/v1/realtime`, PCM16 24 kHz, append/commit/clear | **Xong** |
| R3b — Server VAD / partial | acoustic VAD, partial trước commit | Chưa làm |
| R4 — Capacity | giới hạn production đo trên GPU riêng, scale nhiều GPU | Tooling có, chưa đo |

## 2. Milestone đã hoàn tất

| Milestone | Kết quả | Kiểm chứng |
|---|---|---|
| M0 — Contract + baseline | Pins model/code/SDK, loader offline, fixtures, số đo | [baseline.vi.md](baseline.vi.md), `tests/gpu/test_viegem_parity.py`, `tests/contract/` |
| M1 — Service + scheduler | Config typed, lifespan/health/ready, error envelope, single worker, queue công bằng có giới hạn, logs/metrics | `tests/scheduler/`, `tests/unit/test_app.py`, `tests/gpu/test_engine.py` |
| M2 — Audio ingress | Giới hạn body trước parser, spool có budget, FFmpeg sandbox, chunking tại điểm yên lặng, policy clip ngắn | `tests/audio/`, `tests/unit/test_chunking.py`, `tests/gpu/test_audio_pipeline.py` |
| M3 — REST/SSE + image | Endpoint, joiner, disconnect/send timeout, Dockerfile/compose | `tests/unit/test_transcription_*`, `tests/contract/test_server_sdk.py`, `tests/gpu/test_server.py`, [container-gate.vi.md](container-gate.vi.md) |
| M4 (R2a) — Rich output | Alignment CTC → cue, serializers | `tests/unit/test_alignment.py`, `tests/gpu/test_server.py`, `docs/baseline/alignment_probe.json` |
| M5 — Realtime | Handshake trước accept, session.update, turn buffer + soxr stream, item FIFO, giới hạn session | `tests/unit/test_realtime_*.py`, `tests/gpu/test_realtime.py` |
| M7 (một phần) — Hardening | `scripts/benchmark.py`, OOM/hang/recovery giả lập, soak CPU có giới hạn, redaction log | `tests/unit/test_benchmark.py`, `tests/unit/test_capacity_hardening.py` |

## 3. Bằng chứng hiện có

- **GPU container gate (cu128, RTX 5090):** 6 test pass, không skip; smoke 5 fixture khớp baseline, Realtime,
  auth/lỗi/overload/isolation, SIGTERM khi đang stream → `done` + `drained: true`, restart → ready.
  `docs/baseline/server-container-cu128.json`.
  Bằng chứng này là của QUAN-Transcribe-1.0; chạy lại cho VieGem-ASR ở #5.
- **Server smoke (local GPU, VieGem-ASR):** mọi format + Realtime ở fp32; `vi_long` WER 7.0%; không failure sau policy
  chunk toàn số 0 → text/path rỗng. GPU tier 13 passed. Reports chỉ lưu hash/metadata; WER tổng 6.1% vẫn cao hơn
  baseline QUAN 5.2%; #4 đóng với ngoại lệ owner chấp nhận ([baseline.vi.md](baseline.vi.md) §4–5). `docs/baseline/server.json`.
- **Precision (#6):** fp32 20/329 lỗi, bf16 21/329, fp16 overflow → engine dùng fp32 (không autocast); GPU tier,
  parity fp32/bf16, server smoke và alignment probe chạy lại ở fp32. Cả hai precision còn vượt giới hạn 17/329.
  Trên 349 dev recording ngoài file nguồn fixture: VieGem 9.63%, QUAN 10.33% (thăm dò). Không bật ngưỡng RMS. [baseline.vi.md §8](baseline.vi.md#8-precision-dev-set-lớn-hơn-và-near-zero-a100cu126).
- **CPU suite:** unit + scheduler + audio + contract (Python SDK `openai==3.31.0`, JS SDK `openai@7.33.0`, server
  HTTP/WSS thật) — 447 passed.

## 4. Lệch so với thiết kế ban đầu

- `verbose_json` không có `segments`: schema SDK bắt buộc các field kiểu Whisper (`avg_logprob`, `no_speech_prob`,
  `tokens`, …) mà model không có dữ liệu thật. Cue chỉ có ở `srt`/`vtt`.
- Realtime phát `conversation.item.added` (GA) thay vì `conversation.item.created` (beta).
- Base model `google/embeddinggemma-2` không bị gate trên Hub tại thời điểm staging; không cần `HF_TOKEN`.
- Không dùng `LiveSession` của reference: Realtime chỉ infer khi commit, qua cùng splitter/joiner với REST.
- Thêm profile `cu128` (torch 2.11.0) cho Blackwell vì PyTorch không có wheel 2.14.1 cho CUDA 12.8; `cu126` giữ nguyên.

## 5. Definition of done

Mỗi tính năng phát hành có code + test + tài liệu, cleanup/cancel được kiểm thử, không log dữ liệu nhạy cảm, pins
cố định, capability matrix cập nhật và gate có bằng chứng. Tính năng chưa qua gate thì không bật.
