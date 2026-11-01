# API contract

Payload đang phục vụ: REST/SSE (R1), `verbose_json`/`srt`/`vtt` (R2a), Realtime explicit commit (R3a).

| File | Nội dung |
|---|---|
| [transcriptions.md](transcriptions.md) | `POST /v1/audio/transcriptions`, SSE, timestamps, giới hạn |
| [realtime.md](realtime.md) | WebSocket `/v1/realtime` |
| [errors.md](errors.md) | Envelope lỗi, mã HTTP, endpoint vận hành |

**Nguồn chuẩn:** ví dụ trong `tests/contract/examples/`, được parse bằng SDK thật (`tests/contract/test_python_sdk.py`, `tests/contract/js/contract.test.ts`). Tài liệu khác ví dụ → ví dụ + test đúng.

## SDK đã pin

| | Pin |
|---|---|
| Python | `openai==3.31.0` (commit `86cae507…`) |
| JS | `openai@7.33.0`, TypeScript 5.9.3, Node ≥ 23.6 |
| Realtime | GA schema (`session.type = "transcription"`), không dùng beta `transcription_session.update` |

**Chưa kiểm chứng** với server OpenAI thật: contract dựa trên type SDK và tài liệu OpenAI. Mã lỗi `code` là của gateway này.

## Capability matrix

| Tính năng | Trạng thái |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | Hỗ trợ (R1) |
| `response_format=json\|text`, `stream=true` (SSE) | Hỗ trợ (R1) |
| `response_format=verbose_json\|srt\|vtt` (không stream) | Hỗ trợ (R2a) |
| WS `/v1/realtime`, `turn_detection: null`, append/commit/clear | Hỗ trợ (R3a) |
| `timestamp_granularities[]=word` | 400, chờ gate R2b ([roadmap](../roadmap.md)) |
| `timestamp_granularities[]=segment` | 400: schema SDK cần field chỉ Whisper có; cue chỉ có qua `srt`/`vtt` |
| `server_vad` | 400 `unsupported_turn_detection` (R3b) |
| `semantic_vad`, `noise_reduction`, `audio/pcmu`, `audio/pcma` | Từ chối |
| `prompt` không rỗng, `keywords`, `include[]=logprobs`, `diarized_json`, `known_speaker_*`, `chunking_strategy` | Từ chối 400 |
| Language khác `vi`, auto-detect, translations, speech, chat, WebRTC, SIP, ephemeral secrets | Không hỗ trợ |

Không endpoint nào trả `logprobs`, `avg_logprob`, `no_speech_prob`, confidence hay token id.

## Models

- Id: `viegem-asr`. Alias `whisper-1` chỉ khi `VIEGEM_WHISPER_ALIAS=true`: được chấp nhận ở mọi field `model` và liệt kê riêng trong `/v1/models` (cùng backend). Id khác, kể cả `quan-transcribe-1.0` cũ → 404 `model_not_found`.
- Object: `{"id", "object": "model", "created", "owned_by": "5CD-AI"}`. `created` = commit time của revision weights đã pin (`1792562586`). Ví dụ: `examples/models.json`, `examples/model.json`.

## Kiểm tra contract

```bash
uv run --extra cu126 pytest tests/contract tests/unit
(cd tests/contract/js && npm ci && npm test)
```

`tests/contract/test_server_sdk.py` chạy gateway thật (Uvicorn + FFmpeg, fake engine): payload khớp `examples/`, Python SDK parse được, rồi chạy `js/server.test.ts` với `VIEGEM_TEST_BASE_URL`, `VIEGEM_TEST_API_KEY`, `VIEGEM_TEST_AUDIO_DIR`, `VIEGEM_TEST_WSS_BASE_URL`, `VIEGEM_TEST_CA` (cert tự ký cho `OpenAIRealtimeWS`). `npm test` đơn lẻ bỏ qua các test đó.

**Khi đổi contract:** cập nhật `tests/contract/examples/`, test cả hai SDK, rồi tài liệu này. Khi đổi version SDK: cập nhật pin trong `pyproject.toml` và `tests/contract/js/package.json`. Bỏ field optional thay vì gửi `null`.
