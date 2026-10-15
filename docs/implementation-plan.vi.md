# Kế hoạch triển khai QUAN-Transcribe-1.0

Tài liệu thiết kế: [deployment-design.vi.md](deployment-design.vi.md).

## 1. Kết quả review thiết kế

**Chấp thuận hướng Python QUAN + FastAPI + một inference worker/GPU.** Triển khai theo các gate bên dưới; không quảng cáo tương thích OpenAI đầy đủ.

Đã đối chiếu trực tiếp `references/QUAN-Transcribe/pyproject.toml`, `quan_transcribe/{transcriber,model,features,live,audio}.py` và `references/QUAN-Transcribe-1.0/config.json`:

- `QuanCTC._acts` là mutable state được forward hooks cập nhật: tất cả inference phải serialize.
- `transcribe(streaming=True)` trả text, không phát event; `LiveSession.accept()` forward toàn bộ buffer và có endpoint riêng. MVP Realtime dùng explicit commit, không dùng LiveSession làm state machine OpenAI.
- `transcribe()` tự split, sort và batch toàn bộ input. Không dùng một lời gọi này cho file dài nếu muốn scheduler xen kẽ các request; gateway phải tạo job theo chunk.
- `split_quiet()` không trả absolute offsets và bỏ chunk ≤100 ms. Cần splitter riêng giữ offsets; chính sách clip ngắn phải được test, không vô tình mất phần đuôi.
- `from_pretrained()` không pin riêng base encoder/tokenizer/feature extractor. Offline reproducibility cần cache snapshot đã pin hoặc patch loader có kiểm thử parity.
- `torch>=2.6`, `transformers>=5.19` đúng với reference. Khả năng cài các version này phải kiểm chứng trước; không hạ dependency để vượt gate.
- `sounddevice` là dependency nhưng được import lazy trong `live.py`, không phải import bắt buộc lúc load ASR. Có thể giữ libportaudio trong image đầu để an toàn; chỉ bỏ sau import/container tests.

**Chưa kiểm chứng trong lần review này:** inference thực, quyền HF, dependency availability, peak RAM/VRAM, OpenAI SDK/schema hiện hành, hiệu quả VAD/alignment. SDK commit trong thiết kế là ứng viên pin, không tự coi contract đã được kiểm chứng lại.

### Các điểm cần bổ sung khi implementation

| Điểm | Quyết định triển khai |
|---|---|
| Upload limits | Giới hạn tổng request body ở ingress/ASGI trước multipart parsing, kể cả chunked upload không có Content-Length; thêm giới hạn file/field/temp disk. Chỉ kiểm tra `UploadFile.size` sau parse là không đủ. |
| FFmpeg an toàn | Không dùng client filename làm path/command; subprocess argv, local temp input, protocol allowlist, cấm network và external-reference container/playlist chưa kiểm chứng. Bound stdout/stderr, samples, wall time và số decode process. |
| Deadline inference | Deadline là soft: bỏ result quá hạn nhưng vẫn giữ worker bận tới khi call thật kết thúc. Hard hang cần supervisor restart process; không hứa có thể ngắt CUDA kernel an toàn bằng coroutine cancellation. |
| Fairness | Job file có giới hạn duration; round-robin giữa request/session, ưu tiên live có quota để không làm file starvation. Một job đang chạy không preempt được. |
| Silence/clip ngắn | Empty và too-short có policy riêng dựa trên giới hạn frontend thực. Không dùng amplitude threshold tùy tiện để loại tiếng nói nhỏ; pure-silence test phải kiểm tra output. |
| Errors sau stream headers | Chốt SSE error event theo profile pin. Disconnect không phát success; đã gửi 200 thì không thể đổi thành HTTP 503. |
| Realtime authentication | Bản đầu server-to-server, Bearer header. Browser public dùng proxy/token service riêng, chưa cung cấp ephemeral-secret API. |
| Packaging | Docker smoke là gate của REST release, không đợi hoàn tất timestamps/Realtime mới đóng gói. |

## 2. Phạm vi và thứ tự release

- **R1 — REST/SSE:** models, transcription `json`/`text`, SSE append-only, auth, errors, limits, Docker một GPU; LM tắt mặc định.
- **R2 — Rich outputs:** segments, `verbose_json`, SRT/VTT; word timestamps chỉ công bố sau alignment gate.
- **R3a — Realtime explicit commit:** GA WebSocket transcription subset, PCM16 24 kHz, append/commit/clear; không hứa partial trước commit.
- **R3b — Server VAD:** acoustic VAD và turn lifecycle; stable-prefix incremental decoding là milestone tùy chọn riêng.
- **R4 — Capacity/scale:** benchmark workload thực, replica/GPU và sticky routing nếu cần.

R2 và R3a có thể phát triển song song sau R1; release theo thứ tự trên nếu không có ưu tiên sản phẩm khác. Không đưa native WS, diarization, punctuation/ITN, translations, TTS, chat, WebRTC/SIP vào đường critical của R1.

## 3. Backlog theo milestone

### M0 — Chốt contract và inference baseline (P0, prerequisite)

Trạng thái: **gate đạt** — bằng chứng và giới hạn trong [baseline.vi.md](baseline.vi.md), contract trong [api-contract.vi.md](api-contract.vi.md).

- [x] Ghi source revision/hash của QUAN, QUAN weights, base assets; nếu reference không có Git metadata, dùng manifest SHA256 cho các file code/config/weights cần thiết.
- [x] Xác minh Python 3.11/3.12, wheel torch/CUDA phù hợp driver, transformers yêu cầu và license/HF access. Không tải/gắn token vào repository. *(3.12 đã chạy; 3.11 chưa thử.)*
- [x] Tạo fixture tiếng Việt có quyền sử dụng, kèm transcript tham chiếu; silence, clip ngắn, hai speaker/file khác nhau cho isolation. Tách fixtures nhỏ trong repo và fixtures lớn tải riêng. *(FLEURS không có speaker id: hai file khác nội dung, không chắc khác người nói.)*
- [x] Chạy full-context và streaming-mask trên cùng fixture, đo cold load/warm inference, peak CPU RAM và GPU allocated/reserved; ghi hardware và decoder settings.
- [x] Stage base model/tokenizer/feature extractor và QUAN bằng revision cụ thể; chạy lại với network tắt. Chọn pin-cache hay minimal loader patch; giữ patch ngoài reference và test parity. *(Minimal loader `app/quan_loader.py`.)*
- [x] Pin release/version của OpenAI Python và JS SDK cùng commit schema; xây capability matrix và examples thực sự parse được.
- [x] Chốt payload SSE done/delta/error và Realtime session/item/usage/error; chỉ điền usage có định nghĩa đo được, không bịa token counts.

**Deliverables:** `pyproject.toml`, lockfile phù hợp tooling, `scripts/smoke_inference.py`, `scripts/stage_models.py`, `docs/baseline.vi.md`, `docs/api-contract.vi.md`, asset manifest và fixtures.

**Gate:** inference hợp lệ trên GPU mục tiêu, offline reload thành công, revisions/outputs/metrics lưu lại. Nếu dependency hoặc HF access chưa đạt: dừng integration thật; chỉ tiếp tục scaffold và fake-engine tests, không công bố serving hoạt động.

### M1 — Service foundation và scheduler (P0, sau M0 contract)

Trạng thái: **gate đạt** với fake engine (`tests/scheduler/`, `tests/unit/test_app.py`); engine thật qua worker +
scheduler cho đúng text baseline M0 offline (`tests/gpu/test_engine.py`). Chưa có endpoint transcription (M3).

- [x] `app/config.py`: typed config, API keys riêng HF token, upload/duration/queue/session/decode limits; fail startup nếu config không hợp lệ. Không có key mặc định public. *(Biến `QUAN_*`; biến `QUAN_*` lạ cũng fail để bắt lỗi gõ; key trùng `HF_TOKEN` bị từ chối.)*
- [x] `app/main.py`: lifespan load/warm-up, `/health`, `/ready`, shutdown/drain; readiness false khi model lỗi hoặc worker đang recover. *(Model load trên worker thread nên `/health` trả lời trong lúc load. `/health` 503 khi load/recovery lỗi hoặc inference/warm-up bị treo quá `QUAN_INFER_HANG_SECONDS`: tín hiệu restart cho supervisor. Warm-up chạy chunk dài nhất cho phép để giữ sẵn peak memory.)*
- [x] `app/errors.py`, `app/schemas.py`: OpenAI envelope, request IDs, mapping validation 400/auth 401/model 404/size 413/limit 429/unavailable 503. *(Request id `req_…` ở header `x-request-id`, SDK đọc được; auth là middleware ASGI, áp dụng cả WebSocket handshake.)*
- [x] `app/engine.py`: adapter duy nhất gọi QUAN; load/warm-up/infer trên dedicated single-thread worker. Endpoint không gọi trực tiếp `_forward`, `_run` hoặc model. *(Greedy, full-context, một chunk/call; chặn cứng waveform < 161 mẫu, non-finite, sai dtype/shape. Lỗi inference → 500 rồi worker tự chạy lại warm-up.)*
- [x] `app/scheduler.py`: bounded job count **và tổng buffered bytes/samples**, per-request quota, chunk jobs, fairness; không enqueue sẵn mọi chunk của mọi file dài. *(Lane = request/session. Lane mới quá giới hạn → 429 ngay; submit chờ capacity tối đa `QUAN_QUEUE_TIMEOUT_SECONDS` rồi 429. Mỗi job không vượt `QUAN_MAX_CHUNK_SECONDS`, kể cả live. Round-robin giữa lane; live tối đa `QUAN_LIVE_PRIORITY_RATIO` job liên tiếp khi có file chờ.)*
- [x] Cancel job chưa chạy; running job hoàn tất rồi discard result. Timeout không trả slot sớm; shutdown timeout do supervisor xử lý nếu worker không thoát. *(Job quá `deadline` trả 503 `request_timeout`, kể cả khi đang chạy; giữ worker và budget tới khi call thật kết thúc.)*
- [x] Structured logs/metrics không audio/transcript/token; queue wait, infer duration, decode failures, occupancy, readiness và worker failure. *(JSON log ra stderr; `/metrics` dạng Prometheus, cần API key. Metric decode failures thêm cùng FFmpeg ở M2.)*

**Gate:** fake engine chứng minh max active inference = 1 ở cả REST/live jobs; queue-full, fairness, cancellation và shutdown tests pass. Fake inference bị block phải chứng minh request timeout không tạo inference song song.

### M2 — Audio ingress và chunking (P0, sau M1 skeleton)

Trạng thái: **gate đạt** — `tests/audio/` (FFmpeg thật + Uvicorn thật, CPU), `tests/unit/test_chunking.py`; pipeline
decode → splitter → engine thật trong `tests/gpu/test_audio_pipeline.py`. Chưa có endpoint transcription (M3 dùng
`app.state.audio.receive/decode` và `chunking.split`).

- [x] `app/audio.py`: bounded spool/temp, kiểm tra ingress body trước parser, cleanup success/error/disconnect; filenames do server tạo. *(Body đếm byte trước khi vào `python-multipart`, cả chunked không Content-Length; Content-Length khai báo quá → 413 trước khi đọc body. ≤ 32 part, field ≤ 8 KiB, ≤ 8 header/part; đúng một part `file`. Spool vào thư mục riêng 0700 (`QUAN_SPOOL_DIR`), tên `mkstemp`, tổng `QUAN_MAX_SPOOL_BYTES` cho mọi upload đang chạy → 429; xóa khi thoát context và xóa cả thư mục lúc shutdown.)*
- [x] FFmpeg subprocess ngoài event loop, format/protocol restrictions, decode timeout, bounded concurrency và stderr. Read PCM output theo block, cắt khi vượt max decoded samples thay vì decode xong mới kiểm tra. *(argv, env chỉ có `PATH` (không lộ API key/HF_TOKEN); `-protocol_whitelist file`, demuxer allowlist wav/flac/mp3/mov(mp4,m4a)/ogg/matroska(webm), decoder allowlist độc lập (mọi tổ hợp FFmpeg đọc được, gồm OGG/FLAC và Matroska/AAC); HLS/ffconcat bị từ chối, không mở mạng. Kill + reap ở mọi đường thoát, kể cả cancel. Decode timeout → 400 `invalid_audio`; hết deadline request → 503 `request_timeout`; chờ slot quá `QUAN_QUEUE_TIMEOUT_SECONDS` → 429. Giữ tối đa 4 KiB stderr.)*
- [x] Normalize mono 16 kHz float32, kiểm tra finite samples; không cho client chọn file path hoặc URL. Áp dụng tổng budget cho upload/decode/queue/infer. *(`receive`/`decode` nhận cùng `deadline` monotonic mà M3 truyền tiếp cho `Lane.submit`. Stream bị cụt (mp3/ogg/webm) decode phần đọc được; m4a mất `moov` → `invalid_audio`.)*
- [x] `app/chunking.py`: split tại quiet point, max duration cấu hình, lưu start/end sample; preserve audio coverage hoặc ghi rõ silence removal. Xử lý tail ngắn có chủ đích. *(Spans liền nhau phủ đúng `[0, n)`, không bỏ silence. Cắt ở giữa cửa sổ 50 ms năng lượng thấp nhất trong 5 s cuối trước giới hạn (hòa → cửa sổ muộn nhất); điểm cắt bị giới hạn để chunk cuối > 1,600 mẫu, không bỏ đuôi như `split_quiet`. `QUAN_MAX_CHUNK_SECONDS` ≥ 1.)*
- [x] Policy empty/silence/too-short; test frontend minimum samples thật. R1 không bắt buộc acoustic VAD, có thể dùng quiet splitter deterministic. *(≤ 1,600 mẫu → không có chunk, transcript `""`, không gọi model. Chunk nhỏ nhất được forward (1,601 mẫu) chạy qua frontend thật cho `""`; silence 2 s qua model cho `""`.)*

Metrics thêm: `uploads_total{outcome}`, `decodes_total{outcome}` (khác `ok` là decode failure), `decode_seconds`,
`active_decodes`, `spool_bytes`. Message `request_timeout` đổi thành "Request timed out." vì giờ áp dụng cả
upload/decode.

**Gate:** WAV/FLAC/MP3/M4A/OGG/WEBM và sample-rate/stereo tests; corrupt/truncated/decode-bomb/oversize/timeout không vượt budget, không còn subprocess/temp files. Kiểm tra multipart chunked body và field flooding.

### M3 — REST/SSE và R1 container (P0, sau M1–M2)

Trạng thái: **REST/SSE đạt, container gate chưa đạt.** Thiết kế chi tiết [m3-design.vi.md](m3-design.vi.md).
CPU: `tests/unit/test_transcription_{params,api}.py` (Uvicorn + FFmpeg thật, fake engine), contract với server thật
(`tests/contract/test_server_sdk.py` + `js/server.test.ts`). GPU: `tests/gpu/test_server.py`; smoke
`scripts/smoke_server.py` → `docs/baseline/server.json` (4 fixture ngắn = baseline offline, `vi_long` WER 7.0%).
Container: file đã viết, `uv sync --frozen --no-dev` + smoke đã chạy trên host từ môi trường giống image, nhưng
**chưa build/chạy image** (host không có Docker; D1).

- [x] `app/transcription.py`: `/v1/models`, `/v1/models/{id}`, multipart transcription, model bắt buộc; alias `whisper-1` opt-in và documented. *(Endpoint transcription là ASGI handler riêng, không dùng `StreamingResponse`.)*
- [x] Parse repeated `include[]`, `timestamp_granularities[]`; language chỉ absent/vi, temperature absent/0; reject prompt không rỗng, rich formats và tham số chưa hỗ trợ. Field lạ theo policy trong contract, không silently claim hỗ trợ. *(Thứ tự validate cố định, field đơn lặp → 400; contract §4.)*
- [x] Dùng chung splitter/decoder/joiner cho JSON, text và SSE; full-context từng chunk, separator deterministic. JSON chỉ có field đã có ngữ nghĩa thật. *(`Joiner` duy nhất; `run_chunks` giữ ≤ `QUAN_QUEUE_MAX_JOBS_PER_LANE` chunk trong queue.)*
- [x] Validate request và decode input trước stream headers trong R1; SSE emit từng chunk final ngay khi xong, một done ở cuối. Nối delta bằng done text; lỗi giữa chừng không có done thành công.
- [x] Disconnect dừng decode/enqueue tiếp, bỏ kết quả đang chạy; bounded outbound buffer và send timeout, proxy buffering disabled. *(Watcher `http.disconnect` sau khi đọc xong body, cho cả JSON/text; mỗi send ≤ `QUAN_SEND_TIMEOUT_SECONDS`; `X-Accel-Buffering: no`. Thêm D3: `QUAN_MAX_DECODED_SECONDS` giới hạn audio đã decode giữ trong RAM → 429.)*
- [ ] `Dockerfile`, `.dockerignore`, `compose.yaml`: non-root, minimal copy, pinned code/dependencies, cache volumes, GPU visible, Uvicorn workers=1; model/HF/API secrets không bake vào image. *(Đã viết: base/uv pin digest, FFmpeg pin apt version, user 10001, read-only + tmpfs, key qua compose secret, `stop_grace_period: 70s`; `tests/container/test_image.py`. Chưa kiểm chứng: cần host có Docker + NVIDIA Container Toolkit.)*
- [x] `README.md`: staging, offline runtime, curl/SDK examples, capabilities, limits, reverse proxy/TLS và recovery hướng dẫn. *(Lệnh local đã chạy thử; phần Docker và ví dụ nginx chưa chạy.)*

**Gate R1:** OpenAI Python SDK thật parse JSON và SSE; JS SDK profile được chọn có integration test tương ứng. Container reload offline, `/ready` sau warm-up, auth/error/overload tests pass; GPU isolation giữa hai fixture không trộn transcript. Smoke so với baseline trên cùng chunk/decoder settings, không hứa parity text nếu segmentation khác.

### M4 — Rich output và alignment (P1, sau R1)

Trạng thái: **R2a đạt (`verbose_json`/`srt`/`vtt`); R2b (word timestamps) chưa đạt — chặn bởi bộ annotate tay.**
Thiết kế + số đo P0: [m4-design.vi.md](m4-design.vi.md). CPU: `tests/unit/test_alignment.py`,
`tests/unit/test_transcription_{params,api}.py`, contract (`examples/transcription_verbose*.json`,
`transcription.{srt,vtt}`, cả hai SDK, server thật). GPU: `tests/gpu/test_engine.py` (path == `greedy_decode`),
`tests/gpu/test_server.py` (gate R2a), `scripts/probe_alignment.py` → `docs/baseline/alignment_probe.json`; smoke
`scripts/smoke_server.py` → `docs/baseline/server.json` có cả ba format.

- [x] Result model nội bộ giữ chunk spans, text, duration; `app/alignment.py` có boundary rõ với adapter QUAN. *(`ChunkResult` = text + best path + frames; `run_chunks` yield `(Span, ChunkResult)`; alignment thuần Python, không torch/tokenizer.)*
- [x] Segment subtitles dùng absolute sample offsets, không giả Whisper IDs/probabilities. Chốt optional/required fields theo SDK profile; nếu không có dữ liệu thật để đáp ứng schema, thu hẹp profile. *(`verbose_json` không có `segments` vì schema cần field Whisper; `timestamp_granularities[]=segment` → 400; cue chỉ có ở `srt`/`vtt`.)*
- [x] Kiểm chứng subsampling, feature padding và frame→sample mapping trên fixtures biết mốc tiếng nói. *(P0: `out_lens` = công thức trên 21 độ dài; shift test 100% ± 1 frame; `EMISSION_SHIFT = 0`.)*
- [x] Word alignment theo final transcript, xử lý tokenizer/unit mapping, repeated tokens/blanks, LM khác greedy và transcript không align được. Không gắn greedy spans vào LM text. *(Gateway luôn greedy: path argmax là alignment của chính text; invariant piece/text sai → 500 `alignment_failed`, không chia đều.)*
- [ ] Test monotonicity, range `[0,duration]`, whitespace-unit definition tiếng Việt, cross-format final text; đánh giá alignment bằng fixture annotated và ngưỡng chốt trước khi release. *(Monotonic/range/cross-format: xong (CPU + GPU). Đánh giá word alignment: chưa — cần bộ annotate tay ≥ 200 từ (m4-design §7); `WORD_TIMESTAMPS = False`, `word` → 400.)*

**Gate R2:** segment/subtitle tests pass trước; word timestamps release riêng khi đạt độ chính xác đã thống nhất. Không làm word alignment fail silently thành uniform spacing.

### M5 — Realtime explicit commit (P1, sau R1; độc lập M4)

- [x] `app/realtime.py`: pin GA session schema; session created/update/updated, unknown-event error, model validation và authentication trước accept theo contract.
- [x] PCM16 little-endian mono 24 kHz; base64 validation, giữ byte lẻ qua append packets; stateful soxr 24→16 kHz, flush tail ở commit và reset đúng ở clear.
- [x] Mỗi session có pending audio riêng và committed item immutable. Chốt empty/too-short commit policy, max turn duration và limit committed items pending inference.
- [x] Commit snapshot/resampler rồi tạo buffer mới để nhận turn tiếp; clear chỉ xóa pending. Không đổi format/rate khi pending audio tồn tại nếu chưa có semantics an toàn.
- [x] Emit committed/item/transcription events với event IDs, item IDs, content index, ordering/previous-item linkage và usage đúng pinned schema. Item nhiều chunk vẫn giữ cùng item; completed là transcript authoritative.
- [x] Receiver/sender tách inference, bounded outbound queue, idle/send/session timeouts; session disconnect không hủy GPU ownership sớm.

**Gate R3a:** SDK/schema state-machine tests: nhiều commit trên cùng socket, clear lúc item trước đang infer, bad base64, split samples, flush tail, resample packetized so với reference, item ordering, queue-full, disconnect, hai session isolation. Không quảng cáo word-by-word trước commit hoặc Realtime hội thoại.
*(Đạt trên CPU + GPU: [m5-design.vi.md](m5-design.vi.md) §9, `docs/baseline/server.json`. Container gate chưa chạy:
host không có Docker.)*

### M6 — Acoustic server VAD và partial tùy chọn (P2, sau M5)

- [ ] Chọn/pin VAD, verify license và CPU budget; semantics threshold/prefix padding/silence duration, offsets theo input timeline.
- [ ] Test speech_started/stopped/committed ordering, leading/trailing silence, speech nhỏ, noise và max turn handling; không dùng CTC blank endpoint thay acoustic VAD.
- [ ] Nếu cần partial: expose CTC spans và thiết kế stable-prefix hoặc native snapshot. Chốt final refinement policy trước; không substring diff trên partial mutable.

**Gate R3b:** turn boundaries đạt tiêu chí fixtures và event contract; semantic_vad vẫn reject. Partial chỉ release nếu append-only/final correction policy có integration tests đầy đủ.

### M7 — Capacity và production hardening (P1, bắt đầu từ R1)

- [ ] `scripts/benchmark.py`: REST ngắn/dài, SSE, WS commits và mixed workload; số client tăng dần, cold/warm, LM on/off nếu có.
- [ ] Đo RTF, queue wait p50/p95, first delta/final latency, decode CPU, RAM/VRAM peak, error/overload rates; định nghĩa first-delta từ mốc nào (upload start/end hoặc commit).
- [ ] Chọn default limits từ số đo; ghi maximum tested load, không suy concurrent sessions từ benchmark H100 của tác giả.
- [ ] Soak/restart/OOM/hung-worker tests, proxy timeouts/TLS, disk budget, log redaction và secret rotation; recovery không nhận traffic khi chưa ready.
- [ ] Chỉ sau đo mới làm replica/GPU + sticky WS routing; microbatch REST và KenLM default là optimization riêng có regression tests.

**Gate:** deployment runbook có giới hạn được đo, cách drain/restart và rollback; SLA nếu cần phải chốt workload/hardware trước, không đặt số latency tùy ý.

## 4. Test layout và CI

```text
tests/
  unit/           # config, validation, splitter, resampler, joiner, item state
  scheduler/      # serialized fake engine, fairness, cancel, memory budgets
  contract/       # HTTP/SSE/WS schema + pinned Python/JS SDK
  audio/          # FFmpeg formats, bounds, cleanup, hostile inputs
  gpu/            # baseline parity, concurrent isolation, real cancellation
  container/      # offline load, readiness, non-root, secrets, restart
```

- PR/CPU lane: unit, scheduler, mocked contract và FFmpeg tests; không cần HF credentials hoặc GPU.
- GPU lane có quyền riêng: staged assets, smoke/parity/isolation và container tests; không log secret và không chạy untrusted PR với token.
- Release lane: pinned SDK integration, golden fixtures, benchmark report và capability matrix; release không bật tính năng chưa qua gate.

## 5. Quyết định cần chốt

| Quyết định | Khi nào | Default đề xuất |
|---|---|---|
| SDK/protocol profile và fields chính xác | M0 | Pin Python + JS versions; GA transcription subset |
| Reproducible base loader strategy | M0 | Stage pinned cache trước; patch tối thiểu nếu offline test chưa đủ |
| Upload/duration/queue/PCM/session budgets | M1, refine M7 | Config bắt buộc hữu hạn; số production từ benchmark |
| Gateway auth và browser access | M1/M5 | Bearer server-to-server, browser public ngoài MVP |
| LM và text postprocessing | M0/M4 | Greedy, giữ text nguyên bản; LM opt-in sau đo |
| SSE/JSON segmentation policy | M2 (chốt) | `app/chunking.split` dùng chung; nối chunk bằng một dấu cách, chunk rỗng không sinh dấu cách |
| Timestamp accuracy/word definition | M4 | Whitespace units, annotated acceptance set |
| Ưu tiên R2 hay R3a | Sau R1 | R2 trước; đổi nếu mic client là mục tiêu chính |

Không gán lịch theo tuần trước M0: dependency availability, model access và loader parity là rủi ro lớn nhất. Khi M0 đạt, estimate từng milestone từ spike thực và phân công owner API/engine/audio/ops; một người có thể kiêm vai trò.

## 6. Definition of done

Mỗi milestone có code + tests + documentation, cleanup/cancellation được kiểm thử, không log dữ liệu nhạy cảm, dependencies/assets pin, capability matrix cập nhật và gate có bằng chứng. Kế hoạch này chưa triển khai server và không thay thế benchmark hoặc contract verification thật.
