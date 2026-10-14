# Triển khai QUAN-Transcribe-1.0 tương thích OpenAI

Kế hoạch triển khai và review code reference: [implementation-plan.vi.md](implementation-plan.vi.md).

## 1. Kết luận

**Chọn inference Python gốc (`quan_transcribe`) + một gateway FastAPI mỏng, đóng gói Docker.**

- Không cần port model sang GGUF/C++ hay đưa vLLM vào đường inference.
- REST transcription, SSE và Realtime WebSocket dùng chung một engine, nhưng có adapter giao thức riêng.
- Khởi đầu một process / một GPU / một model instance; session audio riêng cho mỗi kết nối.
- Tham khảo WhisperLiveKit về decode audio, quản lý session, VAD và backpressure; tham khảo audio.cpp về SSE và vòng đời model. Không dùng nguyên server của chúng rồi tuyên bố tương thích OpenAI đầy đủ.

**Không thể cung cấp toàn bộ chức năng OpenAI chỉ bằng model này.** QUAN là ASR tiếng Việt, không phải model dịch, TTS hay hội thoại. Mục tiêu đúng là *tương thích OpenAI Audio Transcriptions và Realtime transcription trong phạm vi công bố*. Nếu cần cả translations, speech, chat/responses hoặc Realtime hội thoại, gateway phải route sang các engine khác.

Đây là thiết kế, chưa phải server đã triển khai hoặc kết quả benchmark. Môi trường kiểm tra có A100 40 GB, nhưng Python hiện thiếu torch, transformers và huggingface_hub; chưa chạy inference/checkpoint.

## 2. Các reference cho biết gì?

| Reference | Điểm hữu ích | Giới hạn với QUAN |
|---|---|---|
| `references/QUAN-Transcribe` | Inference chính thức, file/batch, live session, LM tùy chọn | Chỉ thư viện/CLI, không có HTTP server OpenAI |
| `references/QUAN-Transcribe-1.0` | Model card, config, weights, units, LM | Không phải checkpoint Whisper/Qwen để thay đường dẫn model vào server có sẵn |
| `references/WhisperLiveKit` | FastAPI, FFmpeg, WebSocket `/asr`, REST `/v1/audio/transcriptions`, Deepgram subset | Chưa có backend QUAN; REST hiện không khai báo `stream`; tài liệu tự gọi là compatibility-oriented subset; `model` bị bỏ qua |
| `references/audio.cpp` | Server native, REST/SSE, live ingest PCM, quản lý residency và queue | Chưa tìm thấy QUAN/EmbeddingGemma implementation trong source/spec/tools; phải port frontend + encoder + reader + decoder + converter, kiểm chứng parity |

audio.cpp có SSE `transcript.text.delta` / `transcript.text.done`, nhưng `/v1/audio/transcriptions/live` là extension của audio.cpp, **không phải OpenAI Realtime**. Endpoint details cũng trả sample offsets, không trực tiếp là schema word timestamp tính bằng giây của OpenAI.

WhisperLiveKit thích hợp hơn nếu ưu tiên UI trình duyệt, diarization, translation pipeline và Deepgram. Khi đó cần viết backend adapter QUAN; không có lệnh `wlk --backend quan` hoạt động ngay trong reference hiện tại. Tương thích OpenAI sâu vẫn cần bổ sung gateway.

## 3. Streaming: có, nhưng phân biệt ba lớp

### A. Streaming mask trong model

`asr.transcribe(audio, streaming=True)` dùng reader attention theo chunk 12 encoder frames × 40 ms = **480 ms**. Encoder attention có left context, convolution reader causal.

Tuy nhiên API này vẫn nhận file/waveform hoàn chỉnh, chạy forward và **trả string**, không phải generator, SSE hay WebSocket. `left_chunks` giới hạn attention context, không tự tạo cache hay giảm waveform đầu vào.

### B. Live audio

`LiveSession.accept(block)` nhận mono float32 16 kHz, trả các câu đã kết thúc và cập nhật `session.partial`; `flush()` kết thúc phần còn lại.

Trong implementation hiện tại:

- Mỗi cập nhật gọi `_forward([self.buf], True, ...)` trên **toàn bộ câu đang tích lũy**. Chưa có persistent encoder/KV cache.
- CLI gom audio đủ một chunk rồi cập nhật khoảng 480 ms; `accept()` tự nó không chặn cadence này. Gateway phải gom block, không gọi inference theo từng packet 20 ms.
- Endpoint mặc định 0,8 s dùng khoảng trống CTC blank sau token cuối, không phải acoustic VAD có thể ánh xạ nguyên xi sang `server_vad`.
- Câu dài bị cắt gần điểm yên lặng, giới hạn mặc định 30 s.
- Final chạy lại **full-context**, có KenLM nếu đã nạp; final có thể khác partial.
- 480 ms là kích thước chunk, **không phải SLA latency end-to-end**. Còn thời gian gom audio, queue, inference, network và endpoint.

Do đó live streaming có thật ở cấp nhận audio liên tục, nhưng chi phí tăng theo chiều dài câu. Ưu tiên GPU cho live và đo tải thực tế trước khi chọn số session.

### C. Streaming giao thức

- **File + `stream=true`:** gửi xong file rồi nhận SSE text trong quá trình xử lý. Không phải mic live upload.
- **Realtime transcription:** gửi liên tục JSON/base64 audio qua WebSocket, nhận event theo item/turn.
- **Native WebSocket:** có thể dùng PCM binary 16 kHz và partial snapshot riêng cho UI; thuận tiện nhưng không gọi là OpenAI-compatible.

## 4. Kiến trúc tối giản

```text
OpenAI SDK / curl / client WebSocket
                 |
          HTTPS reverse proxy (khi public)
                 |
           FastAPI gateway
           ├─ auth + validation + limits + OpenAI errors
           ├─ REST/SSE adapter
           ├─ Realtime transcription adapter
           └─ bounded scheduler / single inference worker
                 |
      QuanTranscribe instance dùng chung
           ├─ offline full-context + optional KenLM
           └─ audio/session buffers riêng
                 |
        PyTorch CUDA + cached HF assets
```

Không cần Redis, Triton, Celery hay Kubernetes ở bản đầu. Chỉ thêm scheduler liên process khi tách API và worker để scale.

**Phải serialize toàn bộ model inference.** `QuanCTC` thu hidden states bằng hooks vào biến mutable `self._acts`; gọi đồng thời cùng instance từ nhiều threads có thể trộn dữ liệu. Một inference executor single-thread hoặc queue dedicated worker là phương án dễ kiểm chứng. Mọi đường gọi `_forward`/`_run` đều phải đi qua worker đó; chỉ khóa REST nhưng bỏ quên live session là không đủ.

Event loop chỉ xử lý I/O, không chạy PyTorch/FFmpeg blocking trực tiếp. Không giữ khóa GPU suốt cả kết nối WebSocket. Scheduler cho session live xen kẽ giữa các job file nhỏ, tránh một file dài chiếm GPU toàn bộ thời gian.

Cấu trúc code dự kiến:

```text
app/
  main.py                # lifespan, health/readiness, auth, errors
  engine.py              # model load, serialized inference, limits
  audio.py               # bounded decode, stateful stream resampling
  schemas.py             # OpenAI request/response/event contracts
  transcription.py       # REST + SSE
  realtime.py            # session/item state machine
  alignment.py           # CTC alignment, giai đoạn 2
  native_ws.py           # tùy chọn, partial snapshot cho UI
Dockerfile
compose.yaml
tests/                   # SDK contract, audio edge cases, session isolation
```

## 5. Contract tương thích cần triển khai

| API / tính năng | Phương án | Trạng thái mục tiêu |
|---|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | Schema model OpenAI, id thật, `created`, `owned_by` | Giai đoạn 1 |
| `POST /v1/audio/transcriptions` | Multipart `file`, `model`, language, response_format… | Giai đoạn 1 |
| `json`, `text` | String ASR, content-type đúng | Giai đoạn 1 |
| `stream=true` | SSE delta/done đúng schema | Giai đoạn 1 |
| `verbose_json`, `srt`, `vtt` | Chunk/utterance spans, duration, segment text; kiểm chứng timestamps | Giai đoạn 2 |
| `timestamp_granularities[]=word` | CTC alignment, không chia đều duration theo số từ | Giai đoạn 2 |
| WS `/v1/realtime` transcription | Session update, append/commit/clear, item events | Giai đoạn 3 |
| `diarized_json`, speaker references | Cần diarization + speaker matching engine độc lập | Tùy chọn |
| prompt/keywords/hotword bias | QUAN chưa có decoder prompt API; cần thực sự implement LM/context bias | Không giả vờ hỗ trợ |
| `include[]=logprobs` | CTC confidence không đồng nghĩa logprob token autoregressive OpenAI | Reject khi chưa định nghĩa/kiểm chứng |
| `/v1/audio/translations` | QUAN không dịch tiếng Anh; cần dịch transcript bằng engine khác | Ngoài QUAN thuần |
| `/v1/audio/speech`, chat/responses, Realtime hội thoại | Engine TTS/LLM khác | Ngoài QUAN thuần |

### Model và tham số

- Id chính: `quan-transcribe-1.0`. `/v1/models` không quảng cáo tính năng chưa có.
- `whisper-1` chỉ là **alias opt-in** cho client hard-code id; ghi rõ backend thực là QUAN và đầu ra khác Whisper. Không giả danh toàn bộ id GPT.
- Thiếu/không biết model: trả lỗi có `param="model"`, không âm thầm dùng model mặc định.
- `language`: bỏ trống hoặc `vi`; ngôn ngữ khác báo unsupported. Không coi bỏ trống là QUAN có auto language detection.
- `temperature`: chấp nhận bỏ trống/0 với deterministic decode được tài liệu hóa; giá trị khác báo unsupported, không âm thầm bỏ qua.
- Prompt không rỗng, include logprobs, speaker references, keywords…: báo 400 khi chưa hỗ trợ. Chấp nhận field trong form nhưng bỏ qua không phải tương thích ngữ nghĩa.
- Parse đúng array multipart dạng `timestamp_granularities[]`, `include[]`, kể cả repeated fields; kiểm tra tương quan response_format/stream/granularities.
- Validation sai trả **400 với OpenAI error envelope**, không để FastAPI trả 422 `detail` mặc định. Auth 401, model không tồn tại 404, quá kích thước 413, rate/concurrency limit 429, inference unavailable 503; tài liệu hóa policy.

```json
{"error":{"message":"Unsupported language: en","type":"invalid_request_error","param":"language","code":"unsupported_language"}}
```

### Audio

Dùng FFmpeg để bao phủ WAV, FLAC, MP3, MP4/M4A, MPEG/MPGA, OGG, WEBM; normalize mono 16 kHz float32 trước khi đưa vào QUAN. libsndfile của thư viện không đủ để cam kết tất cả container OpenAI chấp nhận.

Upload vào spool/temp bounded; đặt giới hạn compressed bytes, decoded duration/sample count và decode timeout. Không nhận URL/path từ client để FFmpeg tự truy cập mạng hoặc file server; sandbox decode/protocol allowlist. Xóa temp ở cả success/error/cancel. Silence/clip cực ngắn cần branch riêng để tránh forward waveform rỗng; output đúng schema rỗng, không tự sinh text.

Giới hạn upload cấu hình được; có thể chọn 25 MB cho profile legacy transcription, nhưng kiểm tra lại tài liệu OpenAI của profile mục tiêu thay vì coi một giới hạn đúng cho mọi model/API.

### Timestamp

API gốc chỉ trả text. Để trả timestamp thực sự cần mở rộng decoder giữ frame spans/absolute offsets từ chunking:

1. Giữ start/end sample của mọi chunk, tránh mất offset khi split quiet hoặc bỏ silence.
2. Greedy CTC giữ token và frame emission; encoder khoảng 40 ms/frame nhưng phải kiểm chứng offset/subsampling/feature padding trên audio chuẩn.
3. Ghép token pieces thành đơn vị text đúng tokenizer; công bố cách hiểu `word` với tiếng Việt (đơn vị theo whitespace không nhất thiết là từ ghép ngôn ngữ học).
4. Với KenLM/full-context final khác greedy, dùng forced alignment CTC cho **final transcript**, không gắn spans greedy cũ vào text mới.
5. Segment-level subtitles có thể dùng utterance/chunk spans trước; nói rõ độ chính xác. Khi chưa hỗ trợ word alignment, reject yêu cầu word timestamp.

Không bịa `avg_logprob`, `no_speech_prob`, confidence hay Whisper token ids chỉ để lấp schema. SDK parsing được không đồng nghĩa dữ liệu có ý nghĩa đúng.

## 6. SSE: đơn giản và đúng semantics trước

Cách khởi đầu chắc chắn nhất: chia file tại điểm yên lặng/VAD, decode từng utterance full-context, final text của utterance trở thành append-only delta. Không feed toàn bộ file qua LiveSession chỉ để có SSE, vì sẽ recompute nhiều lần không cần thiết và làm giảm throughput.

```text
Content-Type: text/event-stream

data: {"type":"transcript.text.delta","delta":"xin chào"}

data: {"type":"transcript.text.delta","delta":" mọi người"}

data: {"type":"transcript.text.done","text":"xin chào mọi người"}

```

Gửi separator giữa utterances nhất quán; nối mọi delta phải bằng `done.text`. Flush event ngay, tắt proxy buffering, dừng scheduling khi client disconnect. Validate trước khi gửi headers; sau headers cần event lỗi theo profile đã kiểm chứng, không gửi `done` cho transcript bị cắt do lỗi.

Không thêm `transcript.text.partial` vào profile OpenAI: đây là extension của audio.cpp, không phải event transcription chuẩn trong SDK đã kiểm tra. Không cắt string `partial[len(previous):]`: partial có thể sửa nội dung, nhất là khi final được decode lại full-context/KenLM.

File ngắn có thể chỉ có một delta và một done; vẫn streaming đúng giao thức nhưng không mang nhiều lợi ích latency. SSE không cần token-by-token decoder autoregressive để hợp lệ. Có thêm `[DONE]` hay không phải theo contract SDK/profile đã pin; không sao chép convention Chat Completions mặc định.

## 7. OpenAI Realtime transcription

Pin một phiên bản giao thức trước; không trộn schema beta `transcription_session.update` cũ với GA `session.update` mới. Schema GA đã kiểm tra:

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "quan-transcribe-1.0", "language": "vi"},
        "turn_detection": null
      }
    }
  }
}
```

OpenAI PCM dùng 24 kHz; QUAN dùng 16 kHz. Gateway nhận PCM16 little-endian mono, decode base64 và **resample có state 24→16 kHz**, giữ fractional phase/tail qua packets. Không ép client gửi 16 kHz rồi quảng cáo tương thích OpenAI. G.711 PCMU/PCMA là profile mở rộng, có codec/resampling riêng nếu implement; bản đầu reject rõ ràng.

MVP dễ nhất: `turn_detection=null`, client gửi append/commit/clear. Sau commit decode final, trả `input_audio_buffer.committed`, `conversation.item.created` và transcription delta/completed với `event_id`, `item_id`, `content_index`, usage theo đúng schema được pin. Queue xử lý phải cho phép nhận turn tiếp trong khi turn trước infer, với giới hạn; clear chỉ xóa pending audio, không xóa item đã commit. Connection vẫn dùng được sau mỗi commit.

**LiveSession không thể ánh xạ nguyên xi:** nó tự endpoint và cắt câu dài trong `accept()`, trả final string không có item/offset. Khi turn detection tắt, không được tự biến khoảng nghỉ thành OpenAI committed item. Cần adapter kiểm soát finalization riêng, hoặc MVP chỉ buffer/infer tại explicit commit. Với audio vượt giới hạn turn, trả lỗi hoặc có internal chunking giữ cùng item theo contract, không tự commit âm thầm.

Bổ sung `server_vad` ở bước tiếp theo bằng acoustic VAD có threshold/prefix padding/silence duration đúng cấu hình và phát speech_started/stopped/committed events. Không ánh xạ `LiveSession.endpoint_s` thành đầy đủ OpenAI server VAD. `semantic_vad` cần turn detector khác, không pretend implement.

**Partial trên mic:** muốn realtime delta trước commit thì cần decoder incremental có committed-prefix rõ ràng. Hai lựa chọn:

- Bản đầu: emit OpenAI delta chỉ cho text đã final; UI native dùng partial snapshot riêng. Tương thích event, nhưng không cam kết hiển thị từng từ trước commit.
- Bản nâng cấp: expose CTC unit/frame spans, giữ lại phần cuối chưa ổn định và commit theo chunk; quyết định chế độ final có/không full-context refinement. Không thể vừa cam kết deltas bất biến vừa sửa prefix đã emit bằng LM sau đó. Nếu profile cho phép final correction, client phải dùng completed transcript làm authoritative và cần test riêng.

Realtime ở đây chỉ là **WebSocket transcription subset**, không tự có WebRTC/SDP, ephemeral client secrets, SIP, tool calls hoặc sinh audio response. Nếu client/browser cần các phần đó, triển khai bổ sung hoặc route sang service khác. Không cung cấp API key server dài hạn cho trình duyệt public.

## 8. Đóng gói và vận hành

### Smoke test inference trước khi viết gateway

```bash
# Trong venv Python 3.11/3.12; chọn torch CUDA wheel phù hợp máy host.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install ./references/QUAN-Transcribe
hf auth login
quan-transcribe --device cuda sample.wav
quan-transcribe --device cuda --streaming sample.wav
```

Phải chấp nhận Gemma terms và quyền truy cập `google/embeddinggemma-2` trên Hugging Face trước. `hf` CLI cần được cài nếu môi trường không có; chỉ chạy smoke test bằng audio fixture hợp lệ. QUAN reference yêu cầu `torch>=2.6`, **`transformers>=5.19`**; không tự hạ transformers nếu version đó chưa có trong package index đang dùng. Xác minh wheel khả dụng, khóa version/source commit và chạy parity trước khi build image.

Model load có thể dùng thư mục local:

```python
from quan_transcribe import QuanTranscribe

asr = QuanTranscribe.from_pretrained(
    "references/QUAN-Transcribe-1.0", device="cuda", lm=False
)
```

Local QUAN weights vẫn cần **base encoder + tokenizer + feature extractor**. `model.safetensors` chứa reader và các lớp encoder fine-tune, không đủ để chạy độc lập; `best.pt` không phải file loader này dùng. `revision`/`token` của from_pretrained hiện chỉ được truyền rõ vào download QUAN/LM, không pin/truyền riêng ở tất cả base asset calls. Muốn reproducible offline phải stage/cache và pin base assets riêng hoặc mở rộng loader.

### Container mục tiêu

- CUDA/PyTorch image phù hợp driver, Python 3.11/3.12; thêm FFmpeg, libsndfile và libportaudio do package gốc import sounddevice.
- Dependency gateway: FastAPI, Uvicorn với WebSocket support, python-multipart, soxr; OpenAI SDK dùng trong contract tests.
- COPY chỉ code + config cần thiết. Không COPY toàn bộ references vào image; model repo hiện chứa cả best.pt và các cây reference lồng nhau.
- Volume HF cache/model; tải trước QUAN `config.json`, `units.json`, `model.safetensors`, base assets và `lm/` nếu bật. Runtime có thể offline sau warm-up.
- HF_TOKEN là runtime/staging secret, không bake vào image, không ghi log. API token của gateway là secret khác.
- Một GPU visible và `uvicorn ... --workers 1`. Nhiều workers sẽ nạp nhiều model và không giải quyết đồng bộ hooks giữa requests trên cùng instance.
- Non-root; `/health` liveness khác `/ready` (model và warm-up xong). Deadline decode/infer, bounded request/session queue, max session duration, max buffered PCM, send/idle timeout.
- Nếu cancellation xảy ra khi PyTorch kernel đang chạy, chờ job kết thúc và bỏ result; không giải phóng khóa rồi chạy job mới trên instance đó ngay.
- Không mặc định dùng 600 s padded batch như thư viện cho live mixed workload. Tune batch_seconds theo RAM/VRAM, chia file thành jobs nhỏ, ưu tiên cadence live.
- Thử LM cho offline/final trước, giữ greedy cho partial. KenLM có chi phí build C++ và CPU; đo rồi mới bật mặc định.

QUAN có 356M params theo model card; benchmark 3.713× realtime trên H100 là kết quả công bố cho workload của tác giả, **không phải số concurrent live sessions**. Loader còn nạp base model trước khi lấy audio tower, có thể có peak RAM lớn. Chưa đo nên không hứa một ngưỡng VRAM/throughput cho máy nhỏ. Hiện có A100 40 GB: smoke test trên một GPU trước, không cần tensor parallel 8 GPU.

Giữ output nguyên bản lowercase tiếng Việt, không dấu câu, số viết thành chữ. Punctuation/ITN là postprocessor tùy chọn, version riêng; không âm thầm đổi transcript và timestamps.

## 9. Lộ trình và kiểm thử chấp nhận

1. **Inference baseline:** xác minh dependency, HF access, checkpoint local, WAV fixture, full-context/streaming outputs và GPU memory. Ghi source/model/base revisions.
2. **REST/SSE:** json/text/models/auth/OpenAI errors, FFmpeg formats, bounded scheduling. Test bằng OpenAI Python SDK thật, không chỉ curl.
3. **Rich output:** segments, subtitles và CTC word alignment; test monotonic spans và final text khớp mọi format.
4. **Realtime:** pin GA schema; explicit commit/clear trước, thêm server VAD, rồi incremental stable-prefix nếu cần latency thấp.
5. **Scale:** đo concurrent sessions, queue delay p50/p95, latency first delta/final, RTF, CPU/RAM/VRAM. Sau đó một replica/GPU + sticky WebSocket routing; file microbatch là tối ưu riêng.

Contract tests tối thiểu:

- SDK `client.audio.transcriptions.create(..., model="quan-transcribe-1.0")` và `stream=True`; parsed delta/done đúng types.
- HTTP payload/status/content-type; malformed form, unknown model, prompt không hỗ trợ, non-vi, invalid format/granularities.
- WAV/MP3/M4A/WEBM, stereo, sample rate khác, silence, clip cực ngắn, corrupt audio, file dài, decompression bound.
- Với SSE: `''.join(deltas) == done.text`, final đúng separator tiếng Việt, không success sau timeout/truncation, client disconnect cleanup.
- Với WS: đúng session/item lifecycle, commit tiếp tục được, clear chỉ ảnh hưởng pending, empty commit policy, base64 lỗi, packets split PCM sample, resampler flush tail, invalid rate/format.
- Session isolation: hai client có audio khác không trộn `_acts`, partial hay transcript; GPU call bị cancel không phá lần tiếp theo.
- Queue đầy/send timeout/backpressure, fairness file/live, restart readiness; không lưu audio/token/transcript vào log mặc định.
- Pin cả OpenAI Python/JS SDK được hỗ trợ. Parse schema + state machine, không chỉ kiểm tra một JSON có field text.

## 10. Nguồn đối chiếu

Local:

- `references/QUAN-Transcribe/README.md`, `pyproject.toml`.
- `references/QUAN-Transcribe/quan_transcribe/{transcriber,live,model,features,audio}.py`.
- `references/QUAN-Transcribe-1.0/{README.md,config.json}`.
- `references/WhisperLiveKit/docs/API.md`, `whisperlivekit/{basic_server,core,timed_objects}.py`.
- `references/audio.cpp/docs/asr.md`, `app/server/{README.md,server_protocol.cpp}`.

Đối chiếu thêm nguồn OpenAI qua mạng (không chỉ suy ra từ các server reference):

- https://developers.openai.com/api/docs/guides/realtime-transcription
- https://developers.openai.com/api/docs/guides/speech-to-text
- SDK commit đã kiểm tra: `openai/openai-python@86cae50785fb7b7bf2a5d2383370b214b304acd2`.
- https://github.com/openai/openai-python/blob/86cae50785fb7b7bf2a5d2383370b214b304acd2/src/openai/types/audio/transcription_create_params.py
- Cùng commit: `transcription_text_delta_event.py`, `transcription_text_done_event.py`, `realtime_transcription_session_create_request.py`, `realtime_transcription_session_audio_input.py`, `realtime_audio_formats.py` và conversation input transcription delta/completed events.

Các profile OpenAI khác nhau có thể hỗ trợ tham số và event khác nhau. Chốt target contract/SDK revision khi triển khai, không dùng cụm “OpenAI API full” thay cho capability matrix và integration tests.
