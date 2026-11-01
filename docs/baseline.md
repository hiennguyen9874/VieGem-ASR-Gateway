# Baseline và pins

Số đo nền của model (ngoài gateway) trên vài fixture. Đây là smoke để phát hiện model hỏng, **không phải benchmark capacity** hay đánh giá WER. Report gốc (chỉ hash/metadata): `docs/baseline/*.json`.

## Pins

| Thành phần | Pin |
|---|---|
| Weights `5CD-AI/VieGem-ASR` | `95eae70b21fb0ef640e3ac4ea8cfa2380872ed20` |
| Base `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Code inference | vendored trong `app/viegem/` từ cùng revision; reference read-only `references/VieGem-ASR` |
| Fixtures `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (vi_vn dev, CC BY 4.0) |
| Python / transformers | 3.12.14 / 5.19.0 |
| torch | `cu126`: 2.14.1+cu126 · `cu128`: 2.11.0+cu128 |
| OpenAI SDK | Python `openai==3.31.0`, JS `openai@7.33.0` |

- **Nguồn pin duy nhất:** `assets/manifest.json` (repo, commit, size, SHA256 từng file). Đổi pin: sửa manifest rồi `scripts/stage_models.py --update-manifest`.
- Cả hai repo model không bị gate (không cần `HF_TOKEN`).
- Loader offline `app/viegem_loader.load_viegem` không resolve Hub id. Parity logits (`torch.equal`, fp32 và bf16) với reference: `tests/gpu/test_viegem_parity.py`.

## Fixtures

Tạo lại tất định bằng `scripts/fetch_fixtures.py` (kiểm SHA256 theo `tests/fixtures/manifest.json`).

| File | Nội dung |
|---|---|
| `tests/fixtures/vi_a.wav`, `vi_b.wav` | 2 clip FLEURS (8.88 s, 8.46 s) + transcript tham chiếu |
| `tests/fixtures/silence_2s.wav` | 2 s số 0 |
| `tests/fixtures/short_200ms.wav` | 200 ms cắt từ `vi_a` |
| `fixtures-large/vi_long.wav` (gitignored) | 10 câu nối nhau, 106.62 s |

## Số đo (A100-PCIE-40GB, cu126, fp32, greedy)

Median 5 lần sau warm-up. Report: `docs/baseline/offline.json`.

| Chỉ số | Giá trị |
|---|---|
| Cold load (page cache → GPU) | 5.5 s |
| Inference đầu tiên (8.88 s audio) | 0.9 s |
| Peak CPU RSS / sau load | 4,015 / 1,192 MB |
| GPU sau load (allocated / reserved) | 1,373 / 1,396 MB |
| GPU peak (allocated / reserved) | 2,126 / 2,650 MB |

| Fixture | Audio | Median | WER |
|---|---:|---:|---:|
| `vi_a.wav` | 8.88 s | 48 ms | 0.0% |
| `vi_b.wav` | 8.46 s | 43 ms | 4.0% |
| `vi_long.wav` | 106.62 s | 168 ms | 7.0% |
| `silence_2s.wav` | 2.00 s | 40 ms → `""` | — |
| `short_200ms.wav` | 0.20 s | 39 ms → `""` | — |

- Output tất định. Mỗi lời gọi có sàn ~40–50 ms do kernel launch (~2,440 launch, chỉ ~14 ms compute GPU), nên batch 8 × 8.9 s chỉ mất 60.5 ms.
- Qua gateway, REST cộng thêm decode FFmpeg (~69 ms): REST 8.88 s ≈ 118–141 ms; Realtime commit → completed ≈ 46 ms (`server.json`).

### So với model cũ QUAN-Transcribe-1.0

| Fixture | Từ | VieGem-ASR | QUAN |
|---|---:|---:|---:|
| `vi_a.wav` | 32 | 0 | 2 |
| `vi_b.wav` | 25 | 1 | 1 |
| `vi_long.wav` | 272 | 19 | 14 |
| **Tổng** | 329 | **20 (6.1%)** | **17 (5.2%)** |

Trên fixture, VieGem **kém hơn** QUAN (lỗi chủ yếu ở tên riêng). `smoke_inference.py` giữ ngưỡng 17/329 và trả exit 1; owner chấp nhận ngoại lệ khi đóng #4, không nới ngưỡng.

Trên FLEURS vi_vn/dev lớn hơn (349 recording ngoài file nguồn fixture, 10,044 từ): VieGem 9.63%, QUAN 10.33%, chênh −0.71 điểm, CI95 bootstrap theo sentence id [−1.19, −0.23]. Đây là bằng chứng thăm dò trên dev, không thay thế ngưỡng fixture. Report: `fleurs-dev-*.json`, `fleurs_comparison.json`.

## Precision

| Precision | Lỗi / 329 | Median vi_a / vi_long | Peak VRAM (alloc / reserved) |
|---|---:|---:|---:|
| bf16 (autocast) | 21 (6.38%) | 52 / 64 ms | 2196 / 2722 MiB |
| fp16 | overflow ở inference đầu | — | — |
| **fp32** (mặc định) | **20 (6.08%)** | 47 / 169 ms | 2126 / 2650 MiB |

Chọn fp32 vì WER fixture thấp hơn > 0.1 điểm. Đổi lại file dài chậm ~2.6× (RTF vẫn 0.0016). Cần throughput file dài: `VIEGEM_PRECISION=bf16`. Trên dev hai precision gần như bằng nhau (9.52% / 9.51%). Report: `precision-*.json`.

## Giới hạn frontend và im lặng

| Số mẫu 16 kHz | Gọi model trực tiếp |
|---:|---|
| 1 … 160 | **RuntimeError** (cần ≥ 161 mẫu cho một frame log-mel) |
| 161 … 3,200 | `""` |
| 8,000 (0.5 s) | `"có thể"` |

Frame 40 ms; số frame = `#{k : 640k + 160 < L}` (`alignment_probe.json`).

**Policy gateway:**

- Không bao giờ forward < 161 mẫu (`MIN_FORWARD_SAMPLES`); audio ≤ 1,600 mẫu (100 ms, `MIN_AUDIO_SAMPLES`) → `""` không gọi model.
- Splitter giữ cả phần đuôi ngắn (khác `split_quiet` của reference).
- Model thô sinh từ sai trên chunk toàn số 0 ≥ 1 s (vd. 2 s số 0 → `"vaccine"`). Gateway: nếu **mọi mẫu của chunk bằng 0** thì trả text/đường CTC rỗng (vẫn chạy model để warm-up reserve đủ VRAM).
- **Không dùng ngưỡng RMS/biên độ.** `scripts/probe_silence.py` (37 case, `silence_probe.json`) cho thấy nhiễu khác 0 vẫn có thể sinh từ, nhưng một cutoff đủ để loại nó cũng loại lời nói nhỏ nhận đúng (vd. `vi_b` gain 0.01 vẫn WER 4%). Near-zero noise vì vậy vẫn có thể cho từ sai.

## Tái tạo

```bash
uv sync --extra cu126
uv run --extra cu126 scripts/stage_models.py
uv run --extra cu126 scripts/fetch_fixtures.py
CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$(mktemp -d) uv run --extra cu126 --offline scripts/smoke_inference.py \
    --no-network --out docs/baseline/offline.json
```

`--no-network` đặt `HF_HUB_OFFLINE=1` và chặn mọi kết nối mạng trong process. Lệnh đo precision, silence, FLEURS dev: `docs/agents/commands.md`.

## Reports trong `docs/baseline/`

| File | Nội dung |
|---|---|
| `offline.json` | Baseline inference (1 failure: ngưỡng QUAN) |
| `server.json` | Smoke gateway trên GPU, mọi format + Realtime, 0 failure |
| `server-container-cu128.json` | Smoke trong container gate (còn của QUAN, cần chạy lại) |
| `alignment_probe.json` | Hình học frame CTC, độ trễ spike |
| `precision-{bf16,fp16,fp32}.json` | WER, timing, VRAM theo precision |
| `fleurs-dev-{bf16,fp32,quan}.json`, `fleurs_comparison.json` | FLEURS dev 361 recording + bootstrap |
| `silence_probe.json` | 37 case số 0, noise, lời nói nhỏ |

**Chưa đo:** gate LM (WER, p95 decode mỗi chunk 30 s, RTF), Python 3.11, cold load khi page cache lạnh, tải đồng thời ([roadmap](roadmap.md)).
