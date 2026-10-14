# M0 — Inference baseline QUAN-Transcribe-1.0

Kế hoạch: [implementation-plan.vi.md](implementation-plan.vi.md) (M0). Contract API: [api-contract.vi.md](api-contract.vi.md).

## 1. Kết quả gate

| Gate M0 | Kết quả | Bằng chứng |
|---|---|---|
| Inference hợp lệ trên GPU mục tiêu (A100 40 GB) | Đạt | [`baseline/online.json`](baseline/online.json) |
| Offline reload (HF cache rỗng, chặn mọi socket/DNS) | Đạt, text giống hệt online | [`baseline/offline.json`](baseline/offline.json) |
| Revisions/outputs/metrics lưu lại | Đạt | `assets/manifest.json`, `tests/fixtures/manifest.json`, `uv.lock`, JSON ở trên |
| Loader tự viết khớp loader reference | Đạt: state dict, token map, logits full/streaming `torch.equal` | `tests/gpu/test_loader_parity.py` |
| SDK pin + examples parse được | Đạt: Python 3.31.0, JS 7.33.0 | `tests/contract/` |

Kết quả này chỉ là smoke/baseline trên vài fixture, **không phải benchmark năng lực phục vụ** (M7) và không phải đánh giá WER.

## 2. Revisions đã pin

| Thành phần | Pin | Ghi chú |
|---|---|---|
| Code `quan-transcribe` | `github.com/qnguyen3/QUAN-Transcribe@88ddcd2130a3c355d081bc002439f29e5e849060` | = `references/QUAN-Transcribe` HEAD = `main` lúc kiểm tra; cài qua git pin trong `pyproject.toml`/`uv.lock` |
| Weights `qnguyen3/QUAN-Transcribe-1.0` | `a08726d9fffeb5f16fc1ab05eecc8ad0b86af5a0` | = `references/QUAN-Transcribe-1.0`; SHA256 file khớp bản local và LFS metadata HF. Không stage `best.pt` (loader không dùng) |
| Base `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` | encoder + feature extractor + tokenizer; SHA256 trong manifest |
| Fixtures `google/fleurs` (dataset) | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | vi_vn dev, CC BY 4.0 |
| Python / torch / transformers | 3.12.14 / 2.14.1+cu126 / 5.19.0 | torch cùng version với `uv.lock` của reference, nhưng build CUDA 12.6 |
| OpenAI SDK | Python `openai==3.31.0` (commit `86cae50785fb7b7bf2a5d2383370b214b304acd2`), JS `openai@7.33.0` (gitHead `dd51b11156f920c57dc2e4ad4162cf28a42fe8a1`) | xem api-contract |

`assets/manifest.json` là nguồn pin duy nhất cho model assets (repo id, commit, size, SHA256 từng file). `tests/unit/test_manifests.py` kiểm tra commit QUAN trong `pyproject.toml` khớp manifest.

## 3. Dependency, license, quyền truy cập

- **Python:** chọn 3.12 (`requires-python = ">=3.12,<3.13"`). 3.11 chưa chạy thử.
- **torch:** driver host 535.230.02 (CUDA 12.2 native, có compat 12.8). Wheel PyPI mặc định của torch 2.14.1 không dùng được với driver này nên lấy từ index `pytorch-cu126` (explicit index trong `pyproject.toml`). Wheel `+cu126` có cho cp312.
- **transformers 5.19.0** có trên PyPI, đúng yêu cầu `>=5.19`; không phải hạ version.
- **HF access:** lúc kiểm tra, cả `google/embeddinggemma-2` và `qnguyen3/QUAN-Transcribe-1.0` đều `gated: false`, staging chạy **không cần token**. Điều này khác thiết kế (giả định base bị gate). Trạng thái gate có thể đổi; `stage_models.py` vẫn đọc `HF_TOKEN` từ env nếu có, không ghi/log token.
- **License:** weights QUAN và base đều theo [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (`license: gemma` trong manifest). Người triển khai phải chấp nhận điều khoản này; repo không phân phối weights.
- **sounddevice** vẫn được cài (dependency của quan-transcribe) nhưng chỉ import lazy trong `live.py`; baseline không cần libportaudio.
- **KenLM/LM:** chưa cài `kenlm`, chưa chạy LM. Mọi số liệu là greedy. File `lm/` đã stage + hash để dùng sau.

## 4. Chiến lược loader offline (quyết định)

**Chọn minimal loader ngoài reference:** `app/quan_loader.py::load_quan(models_dir, device)`.

Lý do: `QuanTranscribe.from_pretrained()` đọc base theo `config.json:encoder_id` / `units.json:model_id` = `google/embeddinggemma-2`, resolve theo branch `main` của Hub; `revision` chỉ áp dụng cho repo QUAN. Pin-cache bằng HF cache phụ thuộc `refs/main` trong cache, dễ lệch âm thầm. Loader mới lặp lại đúng từng bước của `from_pretrained` (dùng lại `QuanCTC`, `load_encoder`, `LogMel` của reference, không sửa code reference) nhưng trỏ `encoder_id`/tokenizer vào thư mục base đã stage, và kiểm tra `encoder_id`/`model_id` gốc đúng là `google/embeddinggemma-2`.

Parity (`tests/gpu/test_loader_parity.py`, cần GPU + mạng): test kiểm tra Hub `main` của base vẫn bằng revision đã pin, rồi so state dict, token pieces và logits full/streaming trên `vi_a`, `vi_b` bằng `torch.equal`. Đã pass. Lưu ý: test này để loader reference tải base vào HF cache của user (~1.5 GB).

## 5. Fixtures

`scripts/fetch_fixtures.py` tạo lại mọi file một cách tất định từ FLEURS đã pin và kiểm tra lại SHA256 trong `tests/fixtures/manifest.json` (đã chạy 3 lần, hash không đổi).

| File | Bộ | Nội dung |
|---|---|---|
| `vi_a.wav`, `vi_b.wav` | nhỏ, commit | Bản sao nguyên byte của 2 clip FLEURS khác nội dung (8.88 s, 8.46 s) kèm transcript tham chiếu |
| `silence_2s.wav` | nhỏ, commit | 2 s số 0 |
| `short_200ms.wav` | nhỏ, commit | 200 ms cắt giữa câu của `vi_a` |
| `vi_long.wav` | lớn, `fixtures-large/` (gitignored) | 10 câu dev nối nhau, khoảng lặng 0.3 s, 106.62 s (> 30 s để chạy qua splitter) |

Giới hạn: FLEURS vi_vn không có speaker id, nhãn giới tính toàn bộ là `MALE`. Vì vậy `vi_a`/`vi_b` được đảm bảo **khác nội dung**, không đảm bảo khác người nói. Isolation test dựa trên khác nội dung là đủ để phát hiện trộn transcript.

## 6. Kết quả đo

Môi trường: 1× NVIDIA A100-PCIE-40GB (`CUDA_VISIBLE_DEVICES=3`, GPU riêng lúc đo; các GPU khác trên host có job khác), driver 535.230.02, 64 CPU, Ubuntu 24.04. Decoder: greedy, `use_lm=False`, `chunk_seconds=30`, `batch_seconds=600`, `left_chunks=None`, autocast bf16 (mặc định của reference trên CUDA). Mỗi fixture × mode chạy 5 lần sau warm-up, lấy median. Số trong bảng lấy từ lần **offline**.

| Chỉ số | Giá trị |
|---|---|
| Cold load (đọc file đã nằm trong page cache → model trên GPU) | 4.9 s (online 5.2 s) |
| Lần inference đầu (8.88 s audio) | 0.77 s (online 0.95 s) |
| Peak CPU RSS của process (gồm nạp full base bf16) | 2,537 MB; RSS sau load 1,131 MB |
| GPU sau load: allocated / reserved | 1,361 / 1,384 MB |
| GPU peak trong các lần đo (≤ 106 s audio/lần): allocated / reserved | 2,130 / 2,424 MB |

| Fixture | Audio | Full: median | Full: WER | Streaming: median | Streaming: WER |
|---|---:|---:|---:|---:|---:|
| `vi_a.wav` | 8.88 s | 55 ms | 6.3% | 53 ms | 6.3% |
| `vi_b.wav` | 8.46 s | 56 ms | 4.0% | 54 ms | 16.0% |
| `vi_long.wav` (4 piece, 1 batch) | 106.62 s | 124 ms | 5.2% | 106 ms | 7.0% |
| `silence_2s.wav` | 2.00 s | 53 ms → `""` | — | 51 ms → `""` | — |
| `short_200ms.wav` | 0.20 s | 53 ms → `""` | — | 50 ms → `""` | — |

WER ở đây là word-level edit distance thô so với cột `transcription` của FLEURS trên 3 clip; chỉ dùng để phát hiện model hỏng, không so được với bảng WER của model card.

Các kiểm tra khác (đều đạt, cả online và offline):

- Output tất định qua 5 lần lặp cho mọi fixture/mode.
- Batch `[vi_a, vi_b]` trong một forward cho text giống từng clip chạy riêng (padding không gây trộn).
- Text online = offline cho mọi fixture/mode.

Quan sát: có khoảng **~50 ms sàn mỗi lời gọi** kể cả với silence 2 s hoặc 0.2 s; 106 s audio chỉ mất ~120 ms. Như vậy chi phí lời gọi chi phối với utterance ngắn. Chưa phân tích nguồn gốc (CPU overhead/launch); phải tính tới khi thiết kế cadence live và microbatch (M5/M7).

## 7. Giới hạn frontend và clip ngắn (đầu vào cho policy M2/M5)

`short_probe` gọi `_run` trực tiếp (bỏ qua `split_quiet`) trên đoạn tiếng nói của `vi_a`:

| Số mẫu 16 kHz | `_run` trực tiếp | `transcribe()` |
|---:|---|---|
| 1, 160 | **RuntimeError** (`unfold`: cần ≥ 161 mẫu cho 1 frame log-mel) | `""` |
| 161 … 3,200 | `""` | `""` |
| 8,000 (0.5 s) | `"có thể"` | `"có thể"` |
| 0 (mảng rỗng) | không gọi | `""` |

- Không được forward waveform < 161 mẫu: frontend crash.
- `transcribe()` không crash vì `split_quiet` bỏ mọi piece ≤ 1,600 mẫu (100 ms), kể cả **đuôi** của file dài: phần đuôi ≤ 100 ms bị mất âm thầm. Splitter riêng ở M2 phải xử lý đuôi có chủ đích (gộp vào chunk trước).
- Đề xuất policy (chốt ở M2/M5, xem api-contract): audio đã decode ≤ 1,600 mẫu → trả transcript rỗng, không gọi model. Silence thật trả `""` mà không cần amplitude threshold (đã kiểm tra output).

## 8. Tái tạo

```bash
uv sync                                                    # Python 3.12, torch 2.14.1+cu126, quan-transcribe@88ddcd2
uv run scripts/stage_models.py                             # tải revision đã pin vào ./models (hoặc $QUAN_MODELS_DIR), kiểm SHA256
uv run scripts/fetch_fixtures.py                           # FLEURS đã pin -> tests/fixtures + fixtures-large, kiểm SHA256
CUDA_VISIBLE_DEVICES=3 uv run scripts/smoke_inference.py --out docs/baseline/online.json
CUDA_VISIBLE_DEVICES=3 HF_HOME=$(mktemp -d) uv run --offline scripts/smoke_inference.py --no-network \
    --out docs/baseline/offline.json
uv run pytest                                              # unit + contract (CPU); GPU parity tự skip nếu thiếu GPU/assets/mạng
(cd tests/contract/js && npm ci && npm test)               # JS SDK contract
```

`--no-network` đặt `HF_HUB_OFFLINE=1` và cài `sys.addaudithook` làm fail mọi `socket.connect`/`getaddrinfo` trong process (đã kiểm tra hook chặn được cả `urllib` lẫn `hf_hub_download`). Môi trường này không cho `unshare -n`, nên chưa kiểm tra bằng network namespace/container `--network none`; việc đó thuộc container test M3.

## 9. Sai khác so với thiết kế và việc còn mở

- Base model hiện **không** bị gate (thiết kế giả định có gate).
- Realtime chọn `conversation.item.added` (GA) thay vì `conversation.item.created` như thiết kế viết; lý do ở api-contract §6.
- Chưa đo: LM/KenLM, Python 3.11, cold load khi page cache lạnh, nhiều request đồng thời, thời gian trong container.
- `tests/gpu/test_loader_parity.py` cần mạng; nếu Hub `main` của base đổi, test fail có chủ đích (khi đó parity cần cache base thủ công tại revision cũ).
