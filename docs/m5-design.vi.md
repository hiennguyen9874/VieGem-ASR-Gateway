# M5 — Realtime explicit commit (R3a): thiết kế và kế hoạch

Plan: [implementation-plan.vi.md](implementation-plan.vi.md) §M5. Contract hiện hành: [api-contract.vi.md](api-contract.vi.md) §6
(payload đã pin và đã parse bằng SDK ở M0; tài liệu này chỉ thêm các điểm M0 chưa chốt, liệt kê ở §6).
Trạng thái: **đã triển khai (P0–P4)**; gate R3a có bằng chứng CPU + GPU (§9). Container gate chưa chạy (host không
có Docker, D1). Số đo ở §2 lấy trên host này (CPU); số GPU ở §9.

## 1. Kết quả mong muốn

- `GET /v1/realtime` (WebSocket) cho GA transcription session: `turn_detection: null`, client tự
  `input_audio_buffer.append` / `commit` / `clear`; PCM16 LE mono 24 kHz.
- Mỗi commit thành một item bất biến; transcript chỉ có **sau commit** (delta theo chunk nội bộ đã final, rồi
  `completed` authoritative). Một socket nhận nhiều commit; turn kế tiếp được nhận trong lúc item trước đang infer.
- Dùng lại toàn bộ pipeline đã kiểm chứng: lane `LIVE` của `Scheduler`, `chunking.split`, `run_chunks`, `Joiner`.
  Endpoint không chạm model/worker. Không thêm biến `QUAN_*` mới (các giới hạn M5 đã có trong `app/config.py`).

Ngoài phạm vi: `server_vad`/partial trước commit (M6), Realtime hội thoại (`response.*`), WebRTC/SIP, ephemeral
client secrets, browser trực tiếp, timestamps trong Realtime, `pcmu`/`pcma`.

## 2. Bằng chứng đã kiểm tra

**Thư viện WebSocket — blocker phải gỡ ở P0.** Môi trường hiện tại **không có** `websockets` lẫn `wsproto`
(`import` lỗi). Uvicorn `--ws auto` khi đó không có WS protocol: endpoint WS không thể chạy, và client SDK cũng
không chạy (`openai` Python cần extra `realtime` = `websockets>=13,<16`; JS `OpenAIRealtimeWS` cần peer `ws@^8.21.0`).

**Uvicorn 0.54 (`websockets-sansio`, được `auto` chọn khi có `websockets`):**

- Có extension `websocket.http.response` → `GatewayMiddleware` đã trả 401 envelope trước accept; cùng cơ chế dùng
  cho 400/404/429/503 của handshake.
- Sau mỗi message nhận được, transport `pause_reading()` tới khi app `receive()`: tối đa ~1 message chờ trong
  server, nên RAM vào ≈ `--ws-max-size` (mặc định 16 MiB) mỗi socket; receiver chậm tự tạo backpressure TCP.
- `send()` chờ `writable` (flow control): client không đọc làm `send` treo → bọc `send_timeout_seconds` là đủ
  bound outbound, không cần hàng đợi riêng.
- Shutdown: Uvicorn tự gửi close **1012** cho mọi socket đã accept và đẩy `websocket.disconnect` vào app *trước*
  lifespan shutdown. Không có cơ hội drain item đang chờ qua socket.
- Keepalive ping 20 s / timeout 20 s mặc định.

**SDK đã pin:**

- Python `client.realtime.connect(model=…)` → `ws(s)://…/v1/realtime?model=…`, header `Authorization: Bearer`;
  `http://` base URL → `ws://`.
- JS `OpenAIRealtimeWS` nhận đúng một trong `model`, `intent: "transcription"` (→ `?intent=transcription`),
  `callID`; **luôn ép `wss:`** (kể cả `buildRealtimeURL` tùy chỉnh). Test JS với server thật cần TLS (cert tự ký,
  truyền `ca` qua `options` của `ws`; có `openssl` trên host).
- `completed.usage` bắt buộc (union `tokens | duration`); GA `AudioTranscription` có `delay`, `keywords`,
  `language`, `languages`, `model`, `prompt`; session có `include`.

**soxr 1.1.0 (`ResampleStream`, spike trên host):**

- 72,000 mẫu float32 24 kHz chia 1–4,800 mẫu/packet ngẫu nhiên, `resample_chunk(…, last=True)` ở cuối:
  **bằng tuyệt đối** (`max |Δ| = 0`, cùng độ dài 48,000) với `soxr.resample` one-shot, cả HQ và VHQ.
  `soxr.resample` HQ là đúng hàm `quan_transcribe.audio.to_16k_mono` dùng → reference của gate.
- Sau `last=True` stream từ chối input mới ("Input after last input"); `clear()` đưa về trạng thái mới (output lượt
  sau == one-shot). Tạo stream mới ≈ 54 µs; một packet 20 ms (480 mẫu) ≈ 8.5 µs; delay nội bộ 132 mẫu.
- 241 mẫu 24 kHz → 161 mẫu 16 kHz; 1 mẫu → 1 mẫu (an toàn vì floor 1,600 mẫu của `split` chặn trước model).

**Code hiện có dùng lại được:** `Scheduler.open_lane(JobKind.LIVE)` + ưu tiên live (`live_priority_ratio`);
`run_chunks(lane, wav, spans, deadline, window)`; `Joiner`; `AudioIngress.hold` (budget RAM audio 16 kHz);
`APIError` (`type`/`code`/`message`/`param` dùng cho `error` event, đã ghi trong `app/errors.py`).

## 3. Quyết định

| # | Quyết định | Đề xuất | Lý do / phương án bị loại |
|---|---|---|---|
| B1 | WS stack | Thêm `websockets>=13,<16` vào dependencies (lock bằng `uv.lock`); dev: `openai[realtime]==3.31.0`; JS: `ws` pin exact (thỏa `^8.21.0`). Chạy Uvicorn `--ws websockets-sansio` **tường minh** (Dockerfile, README, `tests/live_server.py`) | `auto` âm thầm tắt WS khi thiếu lib; tường minh thì thiếu lib = lỗi lúc start. `wsproto` thêm một lib khác SDK client |
| B2 | Kích thước message | `--ws-max-size 2097152` (2 MiB ≈ 32 s audio/append; vượt → Uvicorn đóng 1009) | OpenAI cho append tới 15 MiB, nhưng 16 MiB × số session là RAM vào khó bound; client thực gửi 20–200 ms/packet. Ghi trong contract |
| B3 | Handshake | Trước accept, theo thứ tự: auth (middleware, 401) → query: chỉ `model` (phải phục vụ, 404) và `intent` (phải `transcription`, 400); tham số khác (kể cả `call_id`) → 400 `unsupported_parameter` → draining 503 `shutting_down` → chưa ready 503 `model_not_ready` → số session ≥ `QUAN_MAX_SESSIONS` hoặc `open_lane` đầy → 429 `server_busy` | Lỗi không accept rồi mới báo; mở lane trước accept để server đầy không tốn socket. `intent` là cách JS SDK mở transcription session |
| B4 | Một lane `LIVE` / session, giữ suốt session | Session limit (`QUAN_MAX_SESSIONS` ≤ `QUAN_QUEUE_MAX_LANES`, đã validate) chừa lane cho REST | Lane theo item thì mỗi commit có thể 429 khi REST đông |
| B5 | Buffer pending | Resample **khi append** (stateful `ResampleStream` float32, int16/32768), giữ list mảng float32 16 kHz + tối đa 1 byte lẻ. Commit: `resample_chunk(empty, last=True)`, ghép, tạo stream mới. Clear: bỏ mảng, byte lẻ, stream mới | CPU dàn đều theo append (µs/packet), commit O(nối mảng). Output = one-shot (§2). Lưu int16 24 kHz rồi resample ở commit tiết kiệm 25% RAM nhưng dồn CPU vào commit |
| B6 | Byte lẻ ở commit | Bỏ nửa mẫu lẻ (không tính vào usage). Buffer **0 mẫu nguyên** → `input_audio_buffer_commit_empty`, không tạo item, buffer giữ nguyên | Nửa mẫu không có nghĩa; báo lỗi sẽ làm hỏng commit hợp lệ khác |
| B7 | Commit ngắn | > 0 mẫu nhưng sau resample ≤ 1,600 mẫu (≤ 100 ms): **vẫn tạo item**, `completed` với `transcript: ""` và usage thật, không gọi model (đúng contract §6 + policy REST) | Phương án OpenAI thật (lỗi `commit_empty` dưới 100 ms) chưa kiểm chứng được; đổi sau = đổi contract + examples + 2 SDK test |
| B8 | Giới hạn turn | Append làm buffer vượt `QUAN_MAX_TURN_SECONDS` (120 s) → `input_audio_buffer_full`, **toàn bộ append bị bỏ**, buffer không đổi. Turn > `QUAN_MAX_CHUNK_SECONDS` được `split` thành nhiều job **cùng item** (delta mỗi chunk) | Không tự commit âm thầm (design §7). Từ chối cả append giữ buffer nhất quán với những gì client đã nhận ack |
| B9 | Item chờ | Tối đa `QUAN_MAX_PENDING_ITEMS` (4) item chưa terminal; commit thêm → `too_many_pending_items`, buffer **giữ nguyên** (client commit lại sau) | Bound RAM/GPU per session; không mất audio của client |
| B10 | Thứ tự xử lý item | Một task processor / session, FIFO: item N+1 bắt đầu sau khi N terminal. Trong item, `run_chunks` pipelining (`window = QUAN_QUEUE_MAX_JOBS_PER_LANE`) | Completed theo đúng thứ tự commit (contract cho phép lệch nhưng không cần); đơn giản, deterministic. Song song item không tăng throughput vì GPU đã serialize |
| B11 | Budget RAM | Mẫu 16 kHz pending + committed tính vào `AudioIngress.hold` (cùng `QUAN_MAX_DECODED_SECONDS` với REST). Hết budget lúc append → `error` `server_busy` (append bị bỏ). Trả reservation của item khi item terminal **và** không job nào còn chạy trên view của nó; khi session đóng: qua `Lane.release_audio_when_idle` | Một budget RAM toàn cục dễ vận hành. Cần sửa `release_audio_when_idle` nhận nhiều callback (hiện chỉ giữ một) |
| B12 | Deadline item | `commit + QUAN_REQUEST_TIMEOUT_SECONDS`, truyền vào `run_chunks` | Dùng lại semantics soft deadline M1; không thêm config |
| B13 | Lỗi item | `APIError` từ job/submit (`inference_failed`, `model_not_ready`, `request_timeout`, `server_busy`, `shutting_down`) → `…transcription.failed` `{type, code, message}`; session tiếp tục. Delta đã gửi không thu hồi; `failed` là terminal | Contract §6: đúng một trong `completed`/`failed` |
| B14 | Gửi | Một hàm `send_event` (lock + `send_timeout_seconds`), receiver và processor cùng dùng; timeout = client đã đi → đóng session như disconnect. Không có outbound queue | Uvicorn flow control (§2) đã bound; hàng đợi thêm chỉ thêm RAM |
| B15 | Timeout session | Không message client trong `QUAN_SESSION_IDLE_TIMEOUT_SECONDS` **và** không item pending → `error` `idle_timeout` rồi close 1000. Quá `QUAN_MAX_SESSION_SECONDS` → `error` `session_expired` rồi close 1000 (item đang chờ bị bỏ) | Không đóng session khi client đang chờ transcript |
| B16 | Disconnect / shutdown | `lane.close()`: hủy job chờ, bỏ kết quả job đang chạy; GPU/budget giữ tới khi call kết thúc (scheduler đã làm). Shutdown: Uvicorn đóng 1012, item chưa xong bị bỏ — client phải gửi lại | Không thể drain qua socket đã bị Uvicorn đóng (§2); ghi trong README/contract |
| B17 | Validate `session.update` | Tay, whitelist theo contract §6; key lạ → `unknown_parameter` (`param` = đường dẫn chấm, cắt 64 ký tự). Session chỉ đổi khi toàn bộ update hợp lệ. Luật "đổi format khi còn audio" **vô điều kiện thỏa** vì chỉ có một format hợp lệ → không cần code | SDK là dev-dep, không dùng pydantic SDK ở runtime |
| B18 | Event dispatch | JSON không parse được / không phải object / thiếu `type` chuỗi / type không thuộc union client GA / binary frame → `invalid_event`; type GA khác 4 event hỗ trợ → `unsupported_event`. Không đóng socket vì lỗi event | Contract §6 |
| B19 | IDs | `event_<uuid hex>` cho mọi server event; `item_<uuid hex>`; `previous_item_id` = item commit trước trong session hoặc `null`; `error.event_id` = `event_id` client gửi nếu là chuỗi ngắn | Duy nhất không cần state toàn cục |
| B20 | CPU trên event loop | base64 + resample chạy inline trong receiver (≤ 2 MiB/message theo B2). P4 đo; nếu một append lớn > ~5 ms thì chuyển sang `asyncio.to_thread` cho message > ngưỡng | Không tối ưu trước khi đo |

## 4. Kiến trúc

```text
WS handshake ─ GatewayMiddleware (auth, request id) ─ RealtimeEndpoint (B3: query, ready, open_lane LIVE) ─ accept
  │
  ├─ receiver task: receive → parse → dispatch
  │     session.update ─ validate ─ session.updated | error
  │     append ─ base64 ─ TurnBuffer.append (odd byte, ResampleStream, turn cap, hold) ─ (không ack, như OpenAI)
  │     commit ─ TurnBuffer.commit → Item(id, wav16k, usage_s) ─ send committed + item.added ─ processor queue
  │     clear  ─ TurnBuffer.clear ─ cleared
  │
  ├─ processor task: for item in FIFO:
  │     split(wav, max_chunk) → run_chunks(lane, …, deadline) → Joiner → delta* → completed | failed
  │
  └─ send_event (lock + send_timeout) ── dùng chung; mọi exit: lane.close, release hold, cancel tasks
```

### 4.1 `app/realtime.py`

- `TurnBuffer` (thuần, test CPU, không I/O): `append(pcm: bytes)`, `samples` (16 kHz đã có), `seconds_in` (mẫu 24
  kHz nguyên / 24000), `commit() -> (np.ndarray, usage_s)`, `clear()`. Giới hạn turn tính theo mẫu 24 kHz đầu vào
  (đúng con số client gửi). Không biết gì về WebSocket.
- `parse_session_update(session: dict, current: dict) -> dict` (thuần): raise `APIError` (code/param) hoặc trả
  session mới đầy đủ. Session luôn được serialize đầy đủ, không có `null` cho field JS không cho null (contract §6).
- `RealtimeEndpoint` (ASGI thuần như `TranscriptionEndpoint`): handshake (B3), hai task, timeouts (B15), cleanup,
  metrics/log. Đăng ký `app.router.add_websocket_route("/v1/realtime", …)` trong `install` của module.
- `app/main.py`: gọi `install` của realtime; middleware đã chặn auth trước accept, không đổi.
- `app/scheduler.py`: `release_audio_when_idle` giữ nhiều callback (B11). Không đổi gì khác.
- `app/errors.py`: thêm factory cho code mới (§6); message không chứa audio/transcript.

### 4.2 Observability (metadata)

`realtime_sessions` (gauge), `realtime_sessions_total{outcome}` (`closed`, `disconnected`, `send_timeout`,
`idle_timeout`, `session_expired`, `rejected_*`), `realtime_client_events_total{type}` (chỉ type trong union GA,
còn lại `invalid`), `realtime_items_total{outcome}`, `realtime_commit_to_completed_seconds`,
`realtime_audio_seconds_total`. Log một dòng mỗi session/item: số item, giây audio, outcome. Không log audio,
base64, transcript, `prompt`.

## 5. Edge cases phải có test

- Append chia mẫu lẻ: `[1 byte][3 byte][…]` → cùng waveform với gửi một lần; byte lẻ ở commit bị bỏ (B6); clear
  xóa cả byte lẻ.
- Packetization ngẫu nhiên (1 byte … 2 MiB) → `TurnBuffer.commit()` **bằng tuyệt đối** `soxr.resample` one-shot
  (B5); turn thứ hai sau commit/clear cũng bằng one-shot của riêng nó (không rò state).
- Base64 sai/padding sai/`audio` không phải chuỗi → `error` (`invalid_audio` / `invalid_value`), buffer không đổi.
- Commit 0 mẫu → `commit_empty`; commit 1 byte → `commit_empty`; commit 1 mẫu … 100 ms → item + `completed ""`,
  model không bị gọi (fake engine đếm call).
- Turn > 30 s: một item, nhiều delta, `"".join(deltas) == completed.transcript`; turn chạm 120 s → `buffer_full`,
  append bỏ trọn.
- Nhiều commit liên tiếp trên một socket; commit thứ 5 khi 4 item chưa xong → `too_many_pending_items`, buffer giữ;
  commit lại sau khi item xong → thành công.
- Clear trong lúc item trước đang infer: item đó vẫn `completed` đúng transcript; buffer mới rỗng.
- Thứ tự: `committed` → `item.added` → delta* → terminal cho mỗi item; `previous_item_id` nối chuỗi; event/item id
  duy nhất.
- Job lỗi / `request_timeout` / queue đầy → `failed` cho đúng item, item sau vẫn chạy.
- `session.update`: hợp lệ một phần không đổi gì; mỗi field bị từ chối ở contract §6; key lạ; type sai.
- Event không hỗ trợ / JSON sai / binary frame → lỗi, socket vẫn dùng được.
- Handshake: không key 401, `?model=x` 404, `?intent=other` 400, `?call_id=` 400, server đầy 429, chưa ready / draining
  503 — tất cả trước accept, có `x-request-id`.
- Disconnect giữa lúc infer: lane đóng, không gửi gì, không inference song song, budget lane/RAM trả sau khi call
  kết thúc. Client không đọc → send timeout → session đóng.
- Idle timeout không đóng khi còn item pending; session expired đóng có `error`.
- Hai session đồng thời với audio khác nhau: transcript không trộn (fake engine `text_fn` theo nội dung; GPU thật).
- REST + live cùng lúc: max active inference = 1, live không làm file starvation (đã có ở scheduler, thêm test qua
  endpoint).

## 6. Thay đổi contract (làm ở P3, theo quy trình AGENTS: examples → hai SDK test → `api-contract.vi.md`)

- Query `intent=transcription` (JS SDK) được chấp nhận; tham số query khác → 400 trước accept (B3).
- Mã lỗi mới trong `error` event: `unknown_parameter`, `invalid_value`, `server_busy` (budget RAM), `idle_timeout`,
  `session_expired`; `failed.error.code` có thể là `request_timeout`, `server_busy`, `model_not_ready`,
  `shutting_down`, `inference_failed`.
- B6/B7: định nghĩa "buffer rỗng" = 0 mẫu nguyên; commit ≤ 100 ms cho `completed ""`.
- B2 giới hạn message 2 MiB (close 1009); B15/B16 close codes (1000 sau `error`, 1012 khi server restart).
- Append không có ack (như OpenAI): không phát event khi append thành công.
- Examples mới: `realtime_server.jsonl` thêm các error code trên; giữ ví dụ hiện có.

## 7. Kế hoạch (tăng dần, mỗi bước để lại hệ thống dùng được)

| Phase | Nội dung | Hoàn thành khi |
|---|---|---|
| **P0 — Dependencies + handshake spike** | B1: `websockets` (deps), `openai[realtime]` (dev), `ws` (JS, pin exact); `uv lock`; `tests/live_server.py` dùng `ws="websockets-sansio"`, `ws_max_size` = B2. Route WS tối thiểu chỉ accept + `session.created`. Spike: Python SDK `connect(model=…)` qua `ws://`; JS `OpenAIRealtimeWS({intent})` qua `wss://` với cert tự ký (`openssl` trong fixture test, truyền `ca`); handshake 401/404 bằng denial response mà **cả hai SDK** báo được lỗi; đo base64+resample cho message 2 MiB (B20) | Cả hai SDK nhận `session.created` từ server thật; lỗi handshake có status đúng; số đo B20 ghi vào tài liệu này; toàn bộ test hiện có pass |
| **P1 — Phần thuần** | Test trước: `TurnBuffer` (§5 append/odd byte/parity/clear/cap/short), `parse_session_update`, dispatch event (`invalid_event`/`unsupported_event`). `release_audio_when_idle` nhiều callback + test scheduler | Test CPU pass, không cần socket/GPU; parity resample `array_equal` |
| **P2 — Endpoint** | `RealtimeEndpoint`: handshake B3, receiver/processor/send_event, item FIFO + `run_chunks`, B8–B16, metrics/log. `tests/unit/test_realtime_api.py`: Uvicorn thật + fake engine (đếm call, delay, lỗi tiêm), client `websockets` thô cho các case SDK không gửi được (binary, JSON sai, nửa mẫu) | Mọi case §5 (trừ GPU) pass; max active inference = 1 với REST + 2 session đồng thời; không rò task/lane/RAM budget sau disconnect (assert gauge về 0) |
| **P3 — Contract** | §6: examples + `test_python_sdk.py` (union validate) + `contract.test.ts` (`satisfies`); `test_server_sdk.py`: Python SDK `client.realtime.connect` chạy flow update→append→commit×2→clear, so key/type với `realtime_server.jsonl`; `js/server.test.ts`: `OpenAIRealtimeWS` qua TLS, cùng flow; cập nhật `api-contract.vi.md` §2/§6 | Hai SDK thật parse mọi event server phát trên server thật; `npm test` + `pytest tests/contract` pass |
| **P4 — GPU gate + đóng gói** | `tests/gpu/test_realtime.py`: fixture FLEURS 16 kHz → 24 kHz PCM16 (soxr, quantize) gửi packet 20 ms; (a) transcript == engine chạy trên chính waveform 16 kHz đã commit (cùng `split`), (b) WER so reference ≤ WER REST cùng fixture + 2 điểm (resample FFmpeg ≠ soxr nên không đòi bằng text), (c) hai session song song khác fixture không trộn, (d) turn > 30 s nhiều chunk cùng item. `scripts/smoke_server.py` thêm phần realtime → `docs/baseline/server.json`. Dockerfile CMD + README: `--ws websockets-sansio --ws-max-size 2097152`, ví dụ SDK, proxy (`Upgrade`, read timeout ≥ idle timeout), giới hạn và hành vi 1012 | **Gate R3a** (dưới) có bằng chứng; cập nhật plan M5 + AGENTS |

P0 là điều kiện cho mọi bước sau (không có lib WS thì không có endpoint). P1 có thể làm song song với P0.

**Gate R3a** (plan §M5, ánh xạ test): nhiều commit/socket (P2, P3); clear lúc item trước đang infer (P2); base64
sai (P1, P2); split samples + flush tail + resample packetized == reference (P1); item ordering + linkage (P2, P3);
queue-full / too many pending (P2); disconnect (P2); hai session isolation (P2 fake, P4 GPU); SDK state machine
(P3). Không quảng cáo partial trước commit hoặc Realtime hội thoại. Container gate vẫn phụ thuộc host có Docker (D1).

## 8. Rủi ro

- JS SDK chỉ nói `wss:`: client JS bắt buộc qua TLS (proxy hoặc Uvicorn `--ssl-*`); test JS cần cert tự ký.
- Hành vi OpenAI thật (commit < 100 ms, thứ tự event, mã lỗi) chưa kiểm chứng: contract theo type SDK; B7 có thể
  phải đổi nếu client thật dựa vào lỗi `commit_empty`.
- Uvicorn shutdown đóng 1012 ngay: rolling restart làm mất item đang chờ; giảm bằng drain ở tầng LB (ngừng route
  session mới) trước SIGTERM — thuộc M7.
- Text realtime có thể khác REST trên cùng file (FFmpeg swr vs soxr, PCM16 quantize, ranh giới turn do client
  chọn); gate dùng tolerance WER, không parity text.
- Session giữ lane suốt đời: `QUAN_MAX_SESSIONS` × `QUAN_MAX_TURN_SECONDS` × (1 + `QUAN_MAX_PENDING_ITEMS`) giây audio
  có thể cùng giữ RAM (mặc định 16 × 120 × 5 s ≈ 2.6 h ≈ 600 MB float32) — bị `QUAN_MAX_DECODED_SECONDS` (1 h) chặn
  trước (B11); số mặc định chốt ở M7 từ benchmark.
- Receiver bị block bởi send chậm thì không đọc pong: keepalive Uvicorn (20 s) có thể đóng trước send timeout
  (10 s) — chấp nhận, cả hai đều là client chết.

## 9. Kết quả triển khai

Code: `app/realtime.py` (`TurnBuffer`, `parse_session_update`, `validate_event`, `Session`, `RealtimeEndpoint`),
`app/scheduler.py` (`release_audio_when_idle` giữ nhiều callback, chạy đúng một lần theo thứ tự đăng ký),
`app/audio.py` (`AudioIngress.reserve(n)`; `hold(wav)` = `reserve(len(wav))`), `app/errors.py` (factory realtime +
`send_error` ASGI dùng chung cho denial handshake và 401 middleware), `app/main.py` (install route; middleware thêm
`x-request-id` cả khi `websocket.accept` không có `headers`: lỗi thật lộ ra ở handshake SDK đầu tiên).

Lệch so với §3/§4 (đã ghi vào contract §6):

- B3: thêm tham số query lặp → 400 `invalid_value` (trước khi kiểm tra model).
- B11: reservation tính lúc append theo cận trên số mẫu 16 kHz mà append thêm (`samples_bound`), dùng
  `AudioIngress.reserve` thay vì `hold(wav)` (waveform chỉ có lúc commit). Clear trả ngay; item trả khi terminal và
  lane không còn job chạy; session đóng → `release_audio_when_idle`.
- B20: đo ở P0 (CPU host) cho message gần 2 MiB: JSON 5.1 ms + base64 6.5 ms + int16→float 0.66 ms + resample 9.3 ms
  ≈ 21.7 ms p50 (max 32 ms) > ngưỡng 5 ms → JSON parse, base64 decode và resample của message > 256 KiB
  (`OFFLOAD_CHARS`, ≈ 3 ms inline) chạy trong `asyncio.to_thread`; packet 20 ms ≈ 25 µs vẫn inline. Thứ tự event
  không đổi vì receiver vẫn chờ xong mới đọc message kế.
- §4.1: `TurnBuffer` không có `seconds_in` (usage trả từ `commit()`); `parse_session_update` nhận thêm danh sách
  model phục vụ. §4.2: outcome session thực tế là `disconnected` (client close, 1009, 1012), `send_timeout`,
  `idle_timeout`, `session_expired`, `rejected_<code>`, `cancelled`, `internal_error` (không có `closed`).

Test và bằng chứng gate R3a:

- P1 `tests/unit/test_realtime_units.py` (68): byte lẻ, parity packetized == soxr one-shot (`array_equal`), clear,
  cap, short, `session.update`, dispatch theo union client GA của SDK pin.
- P2 `tests/unit/test_realtime_api.py` (25, Uvicorn `websockets-sansio` thật + fake engine, client `websockets` thô):
  handshake denial, nhiều commit/socket, clear khi item trước đang infer, base64 sai, too many pending, budget,
  timeouts, disconnect/send timeout (gauge về 0, budget trả hết), binary/JSON sai, 1009, hai session song song + REST
  (max một inference đồng thời). Ghi chú: client ngừng đọc → session kết thúc ở send timeout, nhưng Uvicorn chỉ đóng
  transport khi buffer gửi xả xong, nên test tự đóng socket sau khi kiểm tra metric.
- P3: Python SDK `client.realtime.connect` (mọi server event validate bằng union pydantic) và JS
  `OpenAIRealtimeWS` qua `wss:` (cert tự ký) chạy update → append (packet chia lẻ mẫu) → commit ×2 → clear → lỗi
  commit rỗng, cùng lỗi handshake 401/404 trên cả hai SDK. `test_python_sdk.py` kiểm tra error trong examples ==
  factory của gateway.
- P4 `tests/gpu/test_realtime.py` (GPU, packet 20 ms, PCM16 24 kHz từ FLEURS qua soxr HQ): (a) transcript == REST
  trên chính waveform 16 kHz đã commit, (b) WER ≤ REST + 2 điểm, (c) hai session song song không trộn, (d) turn
  37.7 s → 2 delta trong một item. `scripts/smoke_server.py` thêm phần Realtime; `docs/baseline/server.json` (server
  local, GPU, `--ws websockets-sansio --ws-max-size 2097152`): một socket, 5 commit; vi_a/vi_b WER = REST
  (6.25% / 4.00%), vi_long 106.6 s: 4 delta, WER 7.35% vs REST 6.99%; commit → completed ≈ 0.06 s (clip ≤ 9 s),
  0.24 s (106.6 s).
- Đóng gói: Dockerfile CMD thêm `--ws websockets-sansio --ws-max-size 2097152`; container gate chạy smoke trong image
  nên tự bao phủ Realtime khi có host Docker. README: ví dụ SDK, nginx `Upgrade`, giới hạn, hành vi 1012.

