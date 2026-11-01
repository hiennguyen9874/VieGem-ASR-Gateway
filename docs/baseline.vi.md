# Baseline inference và pins — VieGem-ASR

Số đo nền của model (ngoài gateway), các pin tái lập và giới hạn frontend. Đây là smoke/baseline trên vài fixture,
**không phải benchmark năng lực phục vụ** và không phải đánh giá WER.

## 1. Pins

| Thành phần | Pin |
|---|---|
| Weights `5CD-AI/VieGem-ASR` | `95eae70b21fb0ef640e3ac4ea8cfa2380872ed20` (`viegem_asr.pt`, README; chưa stage `lm/`) |
| Code inference | vendored trong `app/viegem/` từ cùng revision (`NOTICE`); reference read-only `references/VieGem-ASR` |
| Base `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` (encoder, feature extractor, tokenizer) |
| Fixtures `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (vi_vn dev, CC BY 4.0) |
| Python / transformers | 3.12.14 / 5.19.0 |
| torch | `cu126`: 2.14.1+cu126 · `cu128`: 2.11.0+cu128 |
| OpenAI SDK (contract) | Python `openai==3.31.0`, JS `openai@7.33.0` |

`assets/manifest.json` là nguồn pin duy nhất cho model (repo, commit, size, SHA256 từng file);
`tests/unit/test_manifests.py` kiểm tra manifest chỉ có `embeddinggemma-2` và `viegem-asr`. Đổi pin: sửa manifest
rồi chạy `scripts/stage_models.py --update-manifest`.

**License:** weights VieGem-ASR và code vendored theo CC-BY-NC-SA-4.0 (**chỉ phi thương mại**), cùng
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) cho VieGem-ASR và base; repo không phân phối weights. Lúc
staging, cả hai repo đều không bị gate (không cần `HF_TOKEN`); `stage_models.py` vẫn đọc `HF_TOKEN` nếu có.

## 2. Loader offline

`app/viegem_loader.load_viegem(models_dir, device)` dựng model từ thư mục base đã stage và `viegem_asr.pt` (đọc
`weights_only`), không resolve Hub id; kiểm tra `model_config.encoder_id` và key ngoài encoder. Parity
(`tests/gpu/test_viegem_parity.py`, cần GPU, không cần mạng): cờ `requires_grad`, logits full/streaming bf16
`torch.equal` với code reference ở cùng revision.

## 3. Fixtures

`scripts/fetch_fixtures.py` tạo lại tất định từ FLEURS và kiểm tra SHA256 (`tests/fixtures/manifest.json`).

| File | Nội dung |
|---|---|
| `tests/fixtures/vi_a.wav`, `vi_b.wav` | 2 clip FLEURS khác nội dung (8.88 s, 8.46 s) + transcript tham chiếu |
| `tests/fixtures/silence_2s.wav` | 2 s số 0 |
| `tests/fixtures/short_200ms.wav` | 200 ms cắt giữa câu của `vi_a` |
| `fixtures-large/vi_long.wav` (gitignored) | 10 câu nối nhau, 106.62 s (đi qua splitter) |

FLEURS vi_vn không có speaker id: `vi_a`/`vi_b` chắc chắn khác nội dung, không chắc khác người nói.

## 4. Số đo (A100-PCIE-40GB, cu126, offline)

Greedy, không LM, chunk 30 s (`app/chunking.split`, splitter của gateway), autocast bf16. Median 5 lần sau warm-up. Report: [`baseline/offline.json`](baseline/offline.json)
(text ở đây là chuẩn so sánh cho smoke và GPU tests).

| Chỉ số | Giá trị |
|---|---|
| Cold load (file trong page cache → model trên GPU) | 5.4 s |
| Inference đầu tiên (8.88 s audio) | 1.0 s |
| Peak CPU RSS / RSS sau load | 4,019 MB / 1,191 MB |
| GPU sau load: allocated / reserved | 1,373 / 1,396 MB |
| GPU peak (≤ 4 × 30 s chunk/lần): allocated / reserved | 2,196 / 2,722 MB |

| Fixture | Audio | Full: median | Full: WER | Streaming mask: median | Streaming mask: WER |
|---|---:|---:|---:|---:|---:|
| `vi_a.wav` | 8.88 s | 55 ms | 0.0% | 54 ms | 0.0% |
| `vi_b.wav` | 8.46 s | 54 ms | 4.0% | 54 ms | 8.0% |
| `vi_long.wav` | 106.62 s | 80 ms | 7.4% | 68 ms | 6.6% |
| `silence_2s.wav` | 2.00 s | 52 ms → `"vaccine"` | — | 52 ms → `"vaccine"` | — |
| `short_200ms.wav` | 0.20 s | 52 ms → `""` | — | 54 ms → `""` | — |

WER thô so với transcript FLEURS, chỉ để phát hiện model hỏng. Output tất định qua các lần lặp; batch hai clip cho
text giống chạy riêng. Có sàn ~50 ms mỗi lời gọi, chi phối với utterance ngắn. Gateway chỉ dùng chế độ full-context.
Report ghi 2 failure: `silence_2s.wav` không ra `""` (§5).

**So với baseline QUAN-Transcribe-1.0 đã ghi** (commit `178c1ec`, cùng máy; QUAN cắt `vi_long` bằng `split_quiet`
của reference). Lỗi từ / số từ tham chiếu, full context:

| Fixture | Từ | VieGem-ASR | QUAN-Transcribe-1.0 |
|---|---:|---:|---:|
| `vi_a.wav` | 32 | 0 (0.0%) | 2 (6.3%) |
| `vi_b.wav` | 25 | 1 (4.0%) | 1 (4.0%) |
| `vi_long.wav` | 272 | 20 (7.4%) | 14 (5.2%) |
| Tổng | 329 | 21 (6.4%) | 17 (5.2%) |

Trên 329 từ fixture, VieGem-ASR **kém hơn** baseline QUAN (6.4% so với 5.2%); lỗi chủ yếu ở tên riêng (`byzantine`,
`bhajan`, `lakkha singh`). Không phải do splitter: cắt `vi_long` theo từng câu cho 6.3%, cả file một lần 7.0%,
fp32 7.0%. Model card (8 test set, cùng normalizer) cho FLEURS 7.33 so với 7.88 của QUAN; fixture ở đây quá nhỏ để
kết luận ngược lại, nhưng tiêu chí "không kém QUAN trên fixture" của #4 **chưa đạt**.

**Nguồn gốc sàn ~50 ms** (đo với QUAN tại `8cffbe7`, cùng encoder; chưa profile lại với VieGem-ASR, bảng trên cho
cùng sàn). Mỗi lời gọi bị giới hạn bởi kernel launch, không bởi compute GPU:

| Đo `_forward` (A100, cu126, bf16, median 10 lần) | Giá trị |
|---|---:|
| 1 s / 8.9 s / 30 s audio, batch 1 | 51.7 / 50.4 / 52.5 ms |
| Batch 8 × 8.9 s, một lời gọi | 60.5 ms |
| Kernel launch mỗi lời gọi | ~2,440 |
| Thời gian kernel trên GPU (torch profiler, 8.9 s) | ~14 ms |

Trên gateway, REST còn cộng thêm decode FFmpeg (~69 ms/request trung bình; xem report gốc), nên REST
8.88 s ≈ 125 ms trong khi Realtime commit → completed ≈ 57 ms. Hệ quả cho tải: [roadmap.md](roadmap.md) §1, §4.

## 5. Giới hạn frontend và clip ngắn

| Số mẫu 16 kHz | Gọi model trực tiếp |
|---:|---|
| 1 … 160 | **RuntimeError** (cần ≥ 161 mẫu cho một frame log-mel) |
| 161 … 3,200 | `""` |
| 8,000 (0.5 s) | `"có thể"` |

Đo lại với VieGem-ASR (`offline.json` `short_probe`): cùng sàn 161 mẫu như QUAN (frame log-mel 320 mẫu,
semicausal pad 160), nên `MIN_FORWARD_SAMPLES = 161` và `MIN_AUDIO_SAMPLES = 1600` giữ nguyên. Hình học frame
(`alignment_probe.json`): số frame = `#{k : 640 k + 160 < L}` với mọi độ dài đo, frame 40 ms; dịch audio `s` mẫu
dịch mọi từ đúng `s` (trong 1 frame, 100%).

Policy của gateway: không bao giờ forward waveform < 161 mẫu; audio ≤ 1,600 mẫu (100 ms) → `""` không gọi model.
Silence dài vẫn qua model (không dùng ngưỡng biên độ). Splitter giữ cả phần đuôi ngắn (khác `split_quiet` của
reference, vốn bỏ đuôi ≤ 100 ms).

**Silence → từ sai (VieGem-ASR, chưa xử lý).** Một chunk toàn số 0 hoặc gần 0 dài từ ~1 s cho đúng một unit ở frame
cuối, logits trùng code reference (không phải lỗi gateway); fp32 cho cùng kết quả:

| Chunk (16 kHz) | Text |
|---|---|
| 0.5 s số 0 | `""` |
| 1 s / 1.5 s / 2 s / 3 s / 10 s / 30 s số 0 | `"đó"` / `"nói"` / `"vaccine"` / `"ú"` / `"trị"` / `"trị"` |
| 2 s hằng số 1e-3 / nhiễu Gauss σ = 1e-5 | `"vaccine"` / `"virus"` |
| 2 s nhiễu Gauss σ = 1e-4 hoặc 1e-3 | `""` |
| Lời nói + tới 2 s số 0 ở đầu hoặc cuối; `vi_a` + 1 s số 0 + `vi_b` | không có từ thừa |

Hệ quả: REST/SSE/`srt`/`vtt` và Realtime trả một từ sai cho upload hoặc turn toàn im lặng số, và cho chunk 30 s
toàn im lặng giữa file dài. Gate GPU hiện fail ở 3 assert silence (`tests/gpu/test_audio_pipeline.py`,
`test_server.py`, `test_realtime.py`); cần quyết định policy trước khi phát hành.

## 6. Tái tạo

```bash
uv sync --extra cu126
uv run --extra cu126 scripts/stage_models.py
uv run --extra cu126 scripts/fetch_fixtures.py
CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$(mktemp -d) uv run --extra cu126 --offline scripts/smoke_inference.py \
    --no-network --out docs/baseline/offline.json
```

`--no-network` đặt `HF_HUB_OFFLINE=1` và chặn mọi `socket.connect`/`getaddrinfo` trong process.

## 7. Reports trong `docs/baseline/`

| File | Nội dung |
|---|---|
| `offline.json` | Baseline inference ở trên (chuẩn text cho smoke/GPU tests) |
| `server.json` | Smoke gateway local trên GPU: mọi format + Realtime (1 failure: Realtime silence, §5) |
| `server-container-cu128.json` | Smoke trong GPU container gate (chỉ metadata; còn của QUAN, chạy lại ở #5) |
| `alignment_probe.json` | Hình học frame CTC và độ trễ spike, căn cứ cho timestamps `srt`/`vtt` |

Chưa đo: LM/KenLM, Python 3.11, cold load khi page cache lạnh, tải đồng thời (xem [roadmap.md](roadmap.md)).
