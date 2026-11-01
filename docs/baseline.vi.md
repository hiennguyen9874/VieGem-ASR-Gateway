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
(`tests/gpu/test_viegem_parity.py`, cần GPU, không cần mạng): cờ `requires_grad`, logits full/streaming ở fp32
(precision serving) và bf16 autocast `torch.equal` với code reference ở cùng revision.

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

Greedy, không LM, chunk 30 s (`app/chunking.split`, splitter của gateway), fp32 không autocast (precision serving,
§8). Median 5 lần sau warm-up. Report: [`baseline/offline.json`](baseline/offline.json)
(SHA256 của text UTF-8 là chuẩn so sánh byte-exact cho smoke và GPU tests; không lưu transcript).

| Chỉ số | Giá trị |
|---|---|
| Cold load (file trong page cache → model trên GPU) | 5.5 s |
| Inference đầu tiên (8.88 s audio) | 0.9 s |
| Peak CPU RSS / RSS sau load | 4,015 MB / 1,192 MB |
| GPU sau load: allocated / reserved | 1,373 / 1,396 MB |
| GPU peak (≤ 4 × 30 s chunk/lần): allocated / reserved | 2,126 / 2,650 MB |

| Fixture | Audio | Full: median | Full: WER | Streaming mask: median | Streaming mask: WER |
|---|---:|---:|---:|---:|---:|
| `vi_a.wav` | 8.88 s | 48 ms | 0.0% | 42 ms | 0.0% |
| `vi_b.wav` | 8.46 s | 43 ms | 4.0% | 43 ms | 8.0% |
| `vi_long.wav` | 106.62 s | 168 ms | 7.0% | 169 ms | 6.6% |
| `silence_2s.wav` | 2.00 s | 40 ms → `""` | — | 41 ms → `""` | — |
| `short_200ms.wav` | 0.20 s | 39 ms → `""` | — | 39 ms → `""` | — |

WER thô so với transcript FLEURS, chỉ để phát hiện model hỏng. Output tất định qua các lần lặp; batch hai clip cho
text giống chạy riêng. Có sàn ~40–50 ms mỗi lời gọi, chi phối với utterance ngắn. Gateway chỉ dùng chế độ full-context.
Report ghi 1 failure: WER tổng cao hơn baseline QUAN (bảng dưới). Script kiểm tra ngưỡng 17/329 và trả exit 1;
không nới ngưỡng để che regression. Silence số 0 đã qua policy ở §5.

**So với baseline QUAN-Transcribe-1.0 đã ghi** (commit `178c1ec`, cùng máy; QUAN cắt `vi_long` bằng `split_quiet`
của reference). Lỗi từ / số từ tham chiếu, full context:

| Fixture | Từ | VieGem-ASR | QUAN-Transcribe-1.0 |
|---|---:|---:|---:|
| `vi_a.wav` | 32 | 0 (0.0%) | 2 (6.3%) |
| `vi_b.wav` | 25 | 1 (4.0%) | 1 (4.0%) |
| `vi_long.wav` | 272 | 19 (7.0%) | 14 (5.2%) |
| Tổng | 329 | 20 (6.1%) | 17 (5.2%) |

Trên 329 từ fixture, VieGem-ASR fp32 **kém hơn** baseline QUAN (6.1% so với 5.2%); lỗi chủ yếu ở tên riêng (`byzantine`,
`bhajan`, `lakkha singh`). Đo trước đó: cắt `vi_long` theo từng câu cho 6.3%, cả file một lần 7.0%.
Bf16 autocast (precision serving trước #6) cho 21/329 lỗi (6.4%) (§8).
Model card (8 test set, cùng normalizer) cho FLEURS 7.33 so với 7.88 của QUAN; không dùng số model card để thay
ngưỡng fixture. Bộ dev lớn hơn ủng hộ VieGem (§8). Tiêu chí "không kém QUAN trên fixture" của #4 **chưa đạt**; owner
chấp nhận ngoại lệ khi đóng #4, ngưỡng thực thi 17/329 giữ nguyên.

**Nguồn gốc sàn ~50 ms** (đo với QUAN tại `8cffbe7`, cùng encoder; chưa profile lại với VieGem-ASR, bảng trên cho
cùng sàn). Mỗi lời gọi bị giới hạn bởi kernel launch, không bởi compute GPU:

| Đo `_forward` (A100, cu126, bf16, median 10 lần) | Giá trị |
|---|---:|
| 1 s / 8.9 s / 30 s audio, batch 1 | 51.7 / 50.4 / 52.5 ms |
| Batch 8 × 8.9 s, một lời gọi | 60.5 ms |
| Kernel launch mỗi lời gọi | ~2,440 |
| Thời gian kernel trên GPU (torch profiler, 8.9 s) | ~14 ms |

Trên gateway, REST còn cộng thêm decode FFmpeg (~69 ms/request trung bình; xem report gốc), nên REST
8.88 s ≈ 118–141 ms trong khi Realtime commit → completed ≈ 46 ms (fp32, `server.json`). Hệ quả cho tải: [roadmap.md](roadmap.md) §1, §4.

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

**Silence → từ sai của model thô (VieGem-ASR).** Một chunk toàn số 0 hoặc gần 0 dài từ ~1 s cho đúng một unit ở frame
cuối, logits trùng code reference (không phải lỗi gateway); fp32 cho cùng kết quả:

| Chunk (16 kHz) | Text |
|---|---|
| 0.5 s số 0 | `""` |
| 1 s / 1.5 s / 2 s / 3 s / 10 s / 30 s số 0 | `"đó"` / `"nói"` / `"vaccine"` / `"ú"` / `"trị"` / `"trị"` |
| 2 s hằng số 1e-3 / nhiễu Gauss σ = 1e-5 | `"vaccine"` / `"virus"` |
| 2 s nhiễu Gauss σ = 1e-4 hoặc 1e-3 | `""` |
| Lời nói + tới 2 s số 0 ở đầu hoặc cuối; `vi_a` + 1 s số 0 + `vi_b` | không có từ thừa |

Policy gateway: sau inference, nếu **mọi mẫu của chunk bằng 0**, trả text/đường CTC rỗng, giữ số frame đã đo.
Vẫn chạy model để warm-up/recovery thực sự reserve peak VRAM. Không dùng ngưỡng RMS/biên độ; audio khác 0, kể cả
rất nhỏ, vẫn giữ greedy path nguyên bản. Vì vậy near-zero noise vẫn có thể cho từ sai; chưa có policy lọc nó.

Kiểm chứng lại A100/cu126: GPU tier **13 passed**, gồm silence qua REST/SSE/timed formats, turn im lặng 31 s qua
nhiều chunk rồi lời nói trên cùng session, và nonzero audio rất nhỏ vẫn khớp greedy decode. Server smoke không
còn failure; alignment probe có 26 chunk, không lỗi invariant. WER fixture vẫn chặn phát hành theo #4.

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
| `offline.json` | Baseline inference: hash transcript, WER, timing; 1 failure ở ngưỡng WER |
| `server.json` | Smoke gateway local trên GPU: mọi format + Realtime, 0 failure; chỉ hash/metadata |
| `server-container-cu128.json` | Smoke trong GPU container gate (chỉ metadata; còn của QUAN, chạy lại ở #5) |
| `alignment_probe.json` | Hình học frame CTC và độ trễ spike, căn cứ cho timestamps `srt`/`vtt` |
| `precision-{bf16,fp16,fp32}.json` | Fixture WER, thời gian/RTF, peak VRAM; fp16 ghi lỗi overflow |
| `fleurs-dev-{bf16,fp32,quan}.json` | Cùng 361 recording vi_vn/dev: hash audio/text, lỗi từ và timing |
| `fleurs_comparison.json` | Tổng hợp dev và paired bootstrap theo sentence id |
| `silence_probe.json` | 37 case số 0, noise và lời nói nhỏ; không thêm bộ lọc |

Chưa đo: gate LM/KenLM (`VIEGEM_LM`, #8: WER fixture, p95 decode mỗi chunk 30 s, RTF), Python 3.11, cold load khi
page cache lạnh, tải đồng thời (xem [roadmap.md](roadmap.md)). Thăm dò một lần, không phải bằng chứng gate (A100
host, fp32): decode LM (beam search + forced alignment) của 30 s đầu `vi_long` mất 0.12 s trên một process CPU; LM
giữ nguyên text của `vi_a`/`vi_b` nhưng trả `có` cho `short_200ms.wav` (greedy `""`: thưởng chèn từ beta).

## 8. Precision, dev set lớn hơn và near-zero (A100/cu126)

### Precision (#6): chọn fp32

Cùng staged weights, greedy, splitter 30 s và padded-batch budget của offline baseline; bf16/fp16 là autocast,
fp32 tắt autocast (không lượng tử hoá lại checkpoint). Median 5 lần sau warm-up. Peak VRAM tính cả workload
full-context/streaming của script, không phải capacity mỗi request của server.

| Precision | Lỗi / 329 từ | WER | Median vi_a / vi_long | RTF vi_a / vi_long | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|
| bf16 | 21 | 6.38% | 52.1 / 63.8 ms | 0.00586 / 0.00060 | 2196 / 2722 MiB |
| fp16 | không đo được | — | lỗi ở inference đầu | — | 1432 / 1466 MiB tới lúc lỗi |
| fp32 | 20 | 6.08% | 46.5 / 168.7 ms | 0.00523 / 0.00158 | 2126 / 2650 MiB |

Fp16 fail với `RuntimeError`: không biểu diễn được giá trị bằng Half (overflow), không coi WER là 0.
Theo rule của #1/#6 (WER fixture thấp nhất; chỉ xét tốc độ/VRAM khi chênh ≤ 0.1 điểm), **engine dùng fp32**
(tắt autocast, mặc định `VIEGEM_PRECISION=fp32`): giảm 0.304 điểm phần trăm WER fixture so với bf16 (> 0.1), peak VRAM thấp hơn
~70 MiB. Đổi lại, file dài chậm hơn ~2.6× (vi_long 169 so với 64 ms/106 s audio, RTF vẫn 0.0016); clip ngắn
không chậm hơn ở lần đo này. Trên dev (bảng dưới) hai precision gần như bằng nhau. Operator cần throughput
file dài có thể đặt `VIEGEM_PRECISION=bf16` (autocast CUDA, đổi lấy +1/329 lỗi fixture); `fp16` bị từ chối lúc start.
Kiểm chứng override: server smoke với bf16 0 failure, WER REST `vi_long` 7.35% như report bf16 trước (fp32: 6.99%).

Đo lại sau khi đổi (A100/cu126): `offline.json` fp32 khớp hash transcript của `precision-fp32.json`, 20/329 lỗi, chỉ
fail ngưỡng QUAN 17/329 (ngoại lệ đã chấp nhận ở #4, không nới ngưỡng); GPU tier 13 passed gồm parity fp32 và bf16;
server smoke và alignment probe 0 failure. Image gate cu126/cu128 chưa chạy lại (#5).

### FLEURS vi_vn/dev: 361 recording, 10,373 từ

Revision dataset giữ nguyên như fixture, `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
Cùng file, cùng splitter, greedy full-context, whitespace WER trên cột normalized transcription, không thêm
normalizer. QUAN dùng checkpoint/hash và reference code đã pin của baseline lịch sử; chỉ dùng để so sánh offline,
không khôi phục loader hay model selection trong gateway.

| Mẫu | VieGem bf16 | VieGem fp32 | QUAN bf16 |
|---|---:|---:|---:|
| Toàn dev: 10,373 từ | 987 lỗi (9.52%) | 986 lỗi (9.51%) | 1053 lỗi (10.15%) |
| Bỏ 12 file nguồn fixture: 349 recording, 10,044 từ | 967 lỗi (9.63%) | 967 lỗi (9.63%) | 1038 lỗi (10.33%) |

Trên 349 recording còn lại, chênh lệch VieGem − QUAN là **−0.707 điểm phần trăm**.
Paired bootstrap 10,000 lần, seed 42, gom theo 149 sentence id (không coi các bản đọc cùng câu là độc lập),
cho CI95 bf16 **[−1.185, −0.225]**, fp32 **[−1.194, −0.227]** điểm phần trăm.
Các recording còn lại vẫn có thể cùng sentence id với fixture; đây không phải tập tách biệt theo nội dung.
Đây là bằng chứng thăm dò trên **dev**, không phải test split hay số đo tương đương model card
(split/normalizer khác); không thay thế điều kiện 329 từ của #4.
Timing dev là một lượt, không phải benchmark latency lặp lại. Đo chưa có KenLM.

### Near-zero: chưa đủ cơ sở bật ngưỡng RMS

`scripts/probe_silence.py`: 37 case float32 16 kHz, seed 42; số 0 từ 100 ms tới 30 s, hằng số và Gauss
biên độ 1e-7..1e-3 ở 2/30 s, hai fixture lời nói với gain 1..1e-4. Không lượng tử hoá PCM16 hoặc resample.
Mọi case toàn số 0 cho text rỗng. Noise khác 0 vẫn có thể sinh từ, không đơn điệu theo RMS.

Ví dụ hằng số 2 s RMS 1e-3 sinh một từ, trong khi `vi_b` gain 0.01, RMS 1.37e-4 vẫn giữ WER 4%.
Một cutoff đủ để loại case noise đó cũng loại lời nói nhận được tốt này.
Ở gain 0.001, cùng clip RMS 1.37e-5 vẫn sinh 9 từ (WER 72%); không thể mặc định mọi tín hiệu nhỏ là im lặng.
**Giữ policy chỉ số 0; không thêm RMS gate hay VAD chưa kiểm định.**

Container cu126/cu128 vẫn chưa được kiểm định lại: môi trường đo không có Docker/Toolkit. Các số ở đây không
thay thế image gate #5; cần chạy trên host có Docker và NVIDIA Container Toolkit.
