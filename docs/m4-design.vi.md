# M4 — Rich output và alignment: thiết kế và kế hoạch

Plan: [implementation-plan.vi.md](implementation-plan.vi.md) §M4. Contract hiện hành: [api-contract.vi.md](api-contract.vi.md).
Trạng thái: **P0–P3 xong (R2a: `verbose_json`/`srt`/`vtt`); P4 (R2b words) bị chặn bởi bộ annotate tay (§7).**
Kết quả đo P0 ở §2.1; các quyết định A5/A6 đã chốt theo số đo đó.

## 1. Kết quả mong muốn

- **R2a (segments):** `response_format` = `verbose_json`, `srt`, `vtt`. Thời gian lấy từ đường CTC greedy của chính
  lần inference đã sinh text, cộng absolute sample offset của chunk. Không tốn thêm GPU.
- **R2b (words):** `timestamp_granularities[]=word` với `verbose_json`, chỉ bật sau khi qua alignment gate (§7).
- Mọi format cho cùng một text: `json.text == text == verbose_json.text == SSE done.text == " ".join(cue texts)`.
- Không bịa `avg_logprob`, `no_speech_prob`, `compression_ratio`, token id kiểu Whisper; không chia đều thời gian.

Ngoài phạm vi: LM/forced alignment (gateway không nạp LM), punctuation/ITN, timestamp trong SSE hoặc Realtime,
diarization.

## 2. Bằng chứng đã kiểm tra

**Schema SDK đã pin** (`openai==3.31.0`, `openai@7.33.0`):

- `TranscriptionVerbose`: bắt buộc `duration`, `language`, `text`; tùy chọn `segments`, `words`, `usage`.
- `TranscriptionWord`: `{word, start, end}` — dữ liệu thật đáp ứng được.
- `TranscriptionSegment`: **10 field bắt buộc** gồm `avg_logprob`, `compression_ratio`, `no_speech_prob`, `seek`,
  `temperature`, `tokens`. JS type khai báo bắt buộc (bỏ field → `satisfies` lỗi khi `tsc --strict`); Python SDK
  parse lỏng: field thiếu thành `None` không báo lỗi (đã chạy `construct_type`). QUAN không có dữ liệu thật cho
  `avg_logprob`/`no_speech_prob`, `tokens` của QUAN không phải token Whisper.

**Hình học frame** (`quan_transcribe/features.py`, `transformers/models/gemma4/modeling_gemma4.py`):

- Log-mel: hop 160, frame 320, pad trái 160 → mel frame `i` có tâm tại sample `160·i`.
- Subsampling: hai `Conv2d(k=3, stride=2, padding=1)` → encoder frame `k` có tâm tại mel frame `4k`, tức sample
  `640·k` (40 ms). Receptive field của phần conv: mel `4k−3 … 4k+3`.
- Mask: mel frame hợp lệ khi `160·i + 320 < L + 160`; mask encoder lấy `[::2][::2]` → frame `k` hợp lệ khi
  `640·k + 160 < L` (suy ra từ code; P0 xác nhận bằng `out_lens`).
- Tầng conformer (chunked attention, left context) và reader (full attention, conv causal) không đổi stride nhưng
  model huấn luyện `dynamic_chunk` nên **spike CTC có thể trễ so với âm thanh** — chưa đo.

**Tokenizer/units** (`models/`, `GemmaTokenizer`):

- 14,917 unit (+ blank 0, unk 1 = `vocab_size` 14,919). **Không có byte-fallback piece**; 10,045 piece bắt đầu
  bằng `▁`; không piece nào có `▁` ở giữa; đúng một piece chỉ là `▁`.
- Hệ quả: một từ (đơn vị theo khoảng trắng) = chuỗi piece từ một piece có `▁` tới trước piece có `▁` kế tiếp.

**Greedy = alignment:** đường argmax là đường tốt nhất toàn cục, nên forced alignment (Viterbi) của chính text
greedy cho cùng đường (trừ hòa). Vì gateway luôn greedy, dùng đường argmax là alignment của final transcript, không
"gắn span greedy vào text khác". Nếu sau này bật LM, phải thêm forced alignment trên text LM; invariant §4.3 chặn
trường hợp text và span lệch nhau.

### 2.1 Kết quả P0 (`scripts/probe_alignment.py` → `docs/baseline/alignment_probe.json`)

Đo trên GPU, greedy full-context, fixture đã commit + `vi_long.wav`:

- **(a) Frame geometry:** `out_lens == #{k : 640k + 160 < L}` trên 21 độ dài (161 … 480,000 mẫu), không ngoại lệ.
  GPU test `tests/gpu/test_engine.py` cũng kiểm tra `frames`, và `units == greedy_decode` của package.
- **(b) Shift:** chèn `s` ∈ {160, 320, 640, 1000, 1280, 3200, 4000, 8000} mẫu zero ở đầu `vi_a`/`vi_b`: 100% từ khớp
  có start và end dịch đúng `s` ± 1 frame (40 ms); bội 640 thì lệch 0. Text đổi nhiều nhất 1 từ/33.
- **(c) Trễ spike:** mỗi từ thường chỉ chiếm 1 frame (median 40 ms, p90 40 ms): spike, không phải khoảng phát âm.
  So với mép năng lượng (cửa sổ 10 ms, −20 dB so với cửa sổ to nhất; nền nhiễu FLEURS ≈ −30 dB), ở 9/14
  utterance mép rõ: start từ đầu tiên trễ 160–240 ms sau onset (spike nằm cuối âm tiết đầu), end từ cuối lệch
  −40…+165 ms so với offset. Các utterance còn lại có nhiễu đầu/cuối làm mép năng lượng sai (0.7–1.4 s), không
  dùng được để ước lượng. Ở composite `vi_a|1 s|vi_b|1 s|vi_a|1 s|vi_b` (37.7 s, 2 chunk, cắt ở 29.2 s): không từ
  nào nằm trong khoảng zero, không cue nào vắt qua; từ cuối trước khoảng zero kết thúc sớm hơn mép 0.44–1.0 s và
  từ đầu sau đó bắt đầu muộn hơn mép 0.86–1.16 s (gồm cả silence tự nhiên của clip + trễ spike).
- **(d) Invariant piece/text:** 26 chunk (mọi fixture, shift, composite): 0 lỗi; `tokenizer.decode` khớp ghép piece.
- **(e) unk:** 1/932 unit (0.1%).
- **Khoảng giữa hai từ:** median 160 ms, p90 600 ms; 12.7% ≥ 500 ms (chủ yếu ngắt câu/khoảng zero).

Chốt:

- **A5: `EMISSION_SHIFT = 0`.** Model tương đương tịnh tiến (b) và end không trễ quá 1 frame so với offset; độ
  lệch ở start là vị trí spike trong âm tiết, không phải trễ hằng: một hằng số dịch cả start lẫn end sẽ làm end sớm.
  Hệ quả cho R2b: start của từ có thể trễ ~1 âm tiết — đúng là việc của gate §7 (và, nếu cần, quy tắc kéo dài duy
  nhất được chọn trên bộ dev).
- **A6:** giữ `PAUSE` 0.5 s, 7 s, 80 ký tự: composite cho 8 cue, 2.0–4.1 s, 37–72 ký tự, ranh giới cue đều ở
  khoảng lặng. Tolerance cho gate R2a: mép cue nằm ngoài khoảng zero và cách mép khoảng zero ≤ 1.5 s (max đo được
  1.16 s, gồm silence tự nhiên của clip).

## 3. Quyết định đề xuất

| # | Quyết định | Đề xuất | Lý do / phương án bị loại |
|---|---|---|---|
| A1 | `segments` trong `verbose_json` | **Không phát `segments`.** `timestamp_granularities[]=segment` → 400 `unsupported_parameter`. Segment chỉ có qua `srt`/`vtt` | Phát segment thiếu field làm hỏng JS type và Python trả `None` âm thầm; điền giá trị giả bị cấm. `segments` là optional nên `verbose_json` không có nó vẫn đúng schema |
| A2 | `language` trong `verbose_json` | `"vietnamese"` | Cùng quy ước tên đầy đủ của Whisper `verbose_json` (ví dụ tài liệu OpenAI: `"english"`); type chỉ là `str`. Phương án khác: `"vi"` |
| A3 | Định nghĩa `word` | Dãy ký tự không phải khoảng trắng của final text (âm tiết tiếng Việt, không phải từ ghép) | Đúng như design §5; công bố trong contract |
| A4 | Đơn vị nội bộ | Sample tuyệt đối 16 kHz (int). Đổi sang giây chỉ khi serialize: JSON = `samples / 16000` làm tròn 3 chữ số; SRT/VTT = ms nguyên `round(samples / 16)` | Rounding đơn điệu nên giữ monotonicity; không lệch tích lũy giữa chunk |
| A5 | Frame → sample | Frame `k` chiếm `[640k − 320, 640k + 320)`, cắt vào `[0, len(chunk)]`, cộng `chunk.start`. Một hằng số trễ `EMISSION_SHIFT` (code, không phải config) chỉ được thêm nếu P0 đo được độ trễ ổn định | Không có tham số runtime cho thứ chưa kiểm chứng |
| A6 | Segment (cue) | Cắt cue tại khoảng lặng giữa hai từ ≥ `PAUSE` (đề xuất 0.5 s); giới hạn cue ≤ `MAX_CUE_SECONDS` (7 s) và ≤ `MAX_CUE_CHARS` (80): từ làm vượt giới hạn mở cue mới. Ngoại lệ: một từ tự nó vượt giới hạn được giữ nguyên trong cue riêng, không cắt từ. Cue = `[start từ đầu, end từ cuối]`. Hằng số chốt ở P0 | Ranh giới cue nằm trong khoảng lặng nên chịu được sai số spike; chunk span (≤ 30 s) quá dài cho phụ đề |
| A7 | Bật word timestamps | Hằng số release trong code (`WORD_TIMESTAMPS = False`), lật trong commit có bằng chứng gate; không phải `QUAN_*` | Operator không bật được tính năng chưa qua gate |
| A8 | Khi alignment lỗi | Chỉ format cần timing mới chạy alignment; vi phạm invariant → 500 `alignment_failed` (log metadata), không fallback chia đều. `json`/`text`/SSE không bị ảnh hưởng | Plan: không fail silently |

## 4. Kiến trúc

```text
EngineWorker ── ChunkResult(text, units, first, last, frames) ──> Scheduler/Lane ──> run_chunks
                                                                                      │
                Joiner.add(result.text)  (không đổi: json, text, SSE, verbose_json text)
                app/alignment.py: words(chunk span, result, pieces) -> segments(words) (chỉ verbose_json/srt/vtt)
                app/formats.py: verbose_json / srt / vtt (hàm thuần)
```

### 4.1 Engine (`app/engine.py`)

- `Engine.transcribe(wav) -> ChunkResult` thay cho `str`. `QuanEngine` gọi `_forward` rồi tự lấy argmax path trên
  CPU; `text` vẫn tính bằng `asr.decode_units(units)` y như `_run` hiện tại (text không đổi so với baseline).
- `ChunkResult` (NamedTuple, numpy/tuple nhỏ, không giữ tensor): `text: str`, `units: tuple[int, ...]` (đã collapse,
  bỏ blank, **giữ** unk để khớp path), `first`/`last: tuple[int, ...]` (frame đầu/cuối của mỗi unit),
  `frames: int` (= `out_lens`).
- `pieces: tuple[str, ...]` (unit id → piece; `""` cho blank/unk) lấy từ tokenizer lúc load, expose qua
  `EngineWorker.pieces` khi ready. Alignment không cần tokenizer/torch.
- Scheduler: `Job.future` đổi kiểu `str` → `ChunkResult`; không đổi logic. Realtime (M5) dùng `.text`.

### 4.2 Endpoint (`app/transcription.py`)

- `run_chunks` yield `(Span, ChunkResult)`; `Joiner` vẫn là nơi duy nhất nối text.
- `Params` thêm `granularities: frozenset[str]`. `RESPONSE_FORMATS` thêm `verbose_json`, `srt`, `vtt`.
- `_final` phân nhánh theo format; alignment chạy inline (O(số unit), CPU nhỏ) sau khi đủ mọi chunk.

### 4.3 `app/alignment.py` (thuần Python, test CPU)

`words(span, result, pieces) -> list[Word(text, start, end)]`:

1. Ghép `pieces[u]` cho các unit có text, thay `▁` bằng khoảng trắng, ghi char range của từng unit.
2. **Invariant:** chuỗi ghép `.strip()` == `result.text`; khác → `AlignmentError`.
3. Từ = run ký tự không phải khoảng trắng; span của từ = hợp các frame của unit chồng lên char range đó; ánh xạ
   frame → sample theo A5.
4. Kiểm tra trước khi trả: `" ".join(w.text) == result.text`, `span.start ≤ start < end ≤ span.end`, start không
   giảm. Vi phạm → `AlignmentError`.

`segments(words, audio_samples) -> list[Segment(text, start, end)]` theo A6; `" ".join(seg.text)` == transcript.

### 4.4 Output

- `verbose_json`: `{"text", "language": "vietnamese", "duration", "usage": {"type": "duration", "seconds"}}`, cộng
  `"words": [...]` khi yêu cầu `word` (mảng rỗng nếu không có từ, không `null`). `duration == usage.seconds`.
- `srt`: `n\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n`, đánh số từ 1; không có cue → body rỗng.
- `vtt`: `WEBVTT\n\n` rồi `HH:MM:SS.mmm --> HH:MM:SS.mmm\ntext\n\n`; không có cue → chỉ header.
- `srt`/`vtt`: `200 text/plain; charset=utf-8` (cả hai SDK trả string cho `text|srt|vtt`; kiểm chứng lại ở P3).
- `stream=true` với `verbose_json`/`srt`/`vtt` → 400 `unsupported_parameter` (`param: "stream"`), như contract §4.

### 4.5 Validation (thứ tự cố định mới)

(1) field lạ, (2) field lặp, (3) `model`, (4) `include[]` → 400, (5) giá trị `timestamp_granularities[]` ∉
{`word`, `segment`} → 400 `invalid_value`, (6) `language`, (7) `temperature`, (8) `prompt`, (9) `response_format`,
(10) `stream`, (11) có granularity mà format ≠ `verbose_json` → 400 `unsupported_parameter`
(`param: "timestamp_granularities[]"`), (12) `segment` → 400 (A1); `word` khi `WORD_TIMESTAMPS = False` → 400,
(13) model không phục vụ (404). Giá trị lặp trong granularities được gộp.

## 5. Edge cases phải có test

- Audio ≤ 100 ms / 0 mẫu: không chunk; `verbose_json` text `""`, `duration` thật, `words: []` nếu yêu cầu; srt rỗng;
  vtt chỉ header.
- Chunk text `""` (silence) → không từ, không cue; chunk lân cận vẫn đúng offset.
- Repeat: `a a` liền nhau → một unit; `a ∅ a` → hai unit. Path extraction phải ra đúng `greedy_decode` của package.
- Unit unk (1) và piece `▁` đơn: không sinh ký tự, không tạo từ rỗng.
- Unit ở frame cuối: `end` cắt về `span.end`; `end ≤ duration` luôn đúng; `start < end` (frame hợp lệ nên
  `640k − 320 < L`).
- Từ bị chunk cắt đôi: thành hai từ ở hai chunk (text cũng có dấu cách do `Joiner`) — nhất quán, ghi trong contract.
- Nhiều chunk: offset tuyệt đối, monotonic qua ranh giới chunk.
- Disconnect/deadline/429: giống `json` (không stream); không thêm đường lỗi mới ngoài `alignment_failed`.

## 6. Kế hoạch (tăng dần, mỗi bước để lại hệ thống dùng được)

| Phase | Nội dung | Hoàn thành khi |
|---|---|---|
| **P0 — Spike đo (GPU)** | `scripts/probe_alignment.py` → `docs/baseline/alignment_probe.json`: (a) `out_lens` vs công thức `640k + 160 < L` trên nhiều độ dài; (b) shift test: chèn `s` mẫu im lặng đầu (`s` = bội và không bội 640) → tỉ lệ từ có start dịch đúng `s` ± 1 frame; (c) fixture ghép `vi_a`/`vi_b` với khoảng zero 1 s đã biết vị trí → khoảng cách từ cuối/đầu tới mép khoảng lặng (ước lượng trễ spike); (d) invariant piece/text trên mọi fixture; (e) tần suất unk | Có số đo; chốt A5 (`EMISSION_SHIFT` hoặc 0) và hằng số A6; cập nhật tài liệu này |
| **P1 — Result model** | `ChunkResult`, engine/worker/scheduler/`run_chunks`, `FakeEngine` sinh path tổng hợp nhất quán với text; `pieces` | Toàn bộ test hiện có pass; GPU: text 4 fixture == `docs/baseline/offline.json`, `units` == `greedy_decode` |
| **P2 — `app/alignment.py`** | Viết test CPU trước: path ngẫu nhiên vs `greedy_decode`, repeat/blank/unk/`▁` đơn, invariant hỏng → lỗi, range/monotonic, ghép nhiều chunk, segmentation (pause, max giây, max ký tự, tie) | Test CPU pass, không phụ thuộc torch/tokenizer runtime |
| **P3 — R2a formats** | Params + `verbose_json` (không words) + `srt` + `vtt`; examples mới trong `tests/contract/examples/` + Python/JS SDK tests + `test_server_sdk.py`/`server.test.ts`; `api-contract.vi.md` §2/§4; README; `smoke_server.py` thêm format mới | **Gate R2a:** cross-format text bằng nhau (CPU + GPU); GPU test fixture khoảng lặng đã biết: không cue nào vắt qua khoảng zero ≥ 1 s, mép cue nằm trong `[mép khoảng lặng ± tolerance P0]`, qua ≥ 1 ranh giới chunk (audio > 30 s); range/monotonic; SDK parse cả ba format |
| **P4 — R2b words** | Bộ annotated (§7), `scripts/eval_alignment.py` → report JSON; nối `words` vào `verbose_json`; lật `WORD_TIMESTAMPS` chỉ khi đạt ngưỡng | **Gate R2b:** ngưỡng §7 đạt trên bộ annotated, report được commit; nếu không đạt, `word` tiếp tục 400 |

P1–P3 không cần dữ liệu annotate, có thể release R2a độc lập với P4.

## 7. Alignment gate (R2b) — cần chốt trước P4

**Trạng thái P4:** đã có `scripts/eval_alignment.py` (định dạng annotation trong docstring, ngưỡng dưới đây là hằng
số trong script, scoring test ở `tests/unit/test_alignment_eval.py`) và nhánh `words` trong `verbose_json` sau
`WORD_TIMESTAMPS` (test bằng monkeypatch). **Chưa có bộ annotate** → chưa chạy gate, `WORD_TIMESTAMPS = False`.
Theo P0 (§2.1), start của từ dự kiến trễ ~1 âm tiết so với ranh giới tay, nên gate nhiều khả năng cần quy tắc kéo
dài start (chọn trên `dev`) hoặc không đạt.

- **Bộ annotated:** ranh giới từ (start/end) gán tay (Praat/Audacity label) trên ≥ 200 từ, ≥ 10 utterance FLEURS
  `vi_vn` (CC BY 4.0, cùng revision pin), cả nam/nữ nếu được; lưu `tests/fixtures/alignment/*.json` kèm SHA256 audio.
  Không dùng output của aligner khác làm ground truth.
- **Ngưỡng đề xuất** (chốt trước khi đo, không chỉnh sau): trên mọi từ có trong cả hypothesis và reference (khớp
  theo edit alignment của chuỗi từ), ≥ 90% start và ≥ 90% end lệch ≤ 100 ms, median lệch ≤ 50 ms; 100% từ có span
  hợp lệ, monotonic. Report ghi riêng tỉ lệ từ không khớp (lỗi ASR, không tính vào sai số thời gian).
- Nếu start/end bị lệch hệ thống do spike CTC (P0: start trễ), chỉ được chọn **một** quy tắc kéo dài span (ví dụ tới đầu
  blank run kế tiếp, tối đa N ms) trên bộ dev tách riêng khỏi bộ test gate.

## 8. Rủi ro

- Spike trễ lớn/không ổn định (model train streaming): word gate có thể không đạt; R2a vẫn dùng được vì ranh giới cue
  nằm trong khoảng lặng.
- Chunk cắt giữa từ (khi không có điểm lặng trong 5 s cuối): hai nửa từ; hiếm với audio có ngắt nghỉ.
- Annotate tốn công người; là blocker duy nhất của R2b.
- Đổi kiểu kết quả engine chạm scheduler/worker/test fake: làm ở P1 riêng, chạy đủ test suite trước khi thêm format.
