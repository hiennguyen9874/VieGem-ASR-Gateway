# Gọi API

Quy tắc đầy đủ của từng field: [api/](api/README.md).

## curl

```bash
KEY=$(sudo head -1 secrets/viegem_api_keys)
URL=127.0.0.1:18000/v1

curl -s $URL/models -H "Authorization: Bearer $KEY"
curl -s $URL/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viegem-asr -F file=@tests/fixtures/vi_a.wav
# {"text":"các sản phẩm thủ công có thể được coi là đồ cổ ...","usage":{...}}

# SSE
curl -sN $URL/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viegem-asr -F stream=true -F file=@tests/fixtures/vi_a.wav

# Phụ đề: srt, vtt; hoặc text, verbose_json
curl -s $URL/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viegem-asr -F response_format=srt -F file=@tests/fixtures/vi_a.wav
```

## OpenAI Python SDK

Đã kiểm thử với `openai==3.31.0` (JS: `openai@7.33.0`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="<gateway key>")

with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="viegem-asr").text)

with open("audio.mp3", "rb") as f:
    for event in client.audio.transcriptions.create(file=f, model="viegem-asr", stream=True):
        print(event.type, getattr(event, "delta", None) or getattr(event, "text", ""))
```

## Realtime

Cần `openai[realtime]`. Audio: PCM16 little-endian mono 24 kHz, base64. Một socket nhận nhiều commit; mỗi commit là một item với các `delta` (một mỗi chunk ≤ 30 s) rồi `completed`.

```python
import base64

with client.realtime.connect(model="viegem-asr") as conn:
    conn.recv()                                       # session.created
    for packet in pcm16_24k_packets:                  # bytes, ví dụ 20–200 ms mỗi packet
        conn.input_audio_buffer.append(audio=base64.b64encode(packet).decode())
    conn.input_audio_buffer.commit()
    for event in conn:
        if event.type == "conversation.item.input_audio_transcription.completed":
            print(event.transcript)
            break
```

## Audio

- Định dạng: WAV, FLAC, MP3, MP4/M4A (AAC/ALAC), OGG (Vorbis/Opus/FLAC), WEBM/Matroska; nhận diện theo nội dung, không theo đuôi.
- Audio dài được cắt thành chunk ≤ 30 s tại điểm yên lặng; SSE gửi một delta mỗi chunk.
- Audio ≤ 100 ms → `""`. Chunk toàn số 0 → rỗng.
- `srt`/`vtt`/`verbose_json` không stream (`stream=true` → 400).
- Alias `whisper-1` (bật bằng `VIEGEM_WHISPER_ALIAS=true`) chỉ để client cũ chạy được: vẫn là VieGem-ASR.
