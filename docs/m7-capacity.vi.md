# M7 — Benchmark tải thực và chốt giới hạn

## 1. Trạng thái và gate

**Tooling/CPU hardening đã triển khai; gate capacity production CHƯA đạt.** Không suy số session từ H100
của tác giả hoặc fake engine. Khi kiểm tra host, cả hai RTX 5090 đều có tiến trình khác sử dụng; chưa có GPU
riêng để đo tải thật. Không thay default limits bằng số đo CPU giả. M6 không phải dependency của M7.

Artifacts:

- `scripts/benchmark.py`: closed-loop tăng số client, REST ngắn/dài, SSE ngắn/dài, WS nhiều explicit commit
  trên socket tái sử dụng; mixed chia đều 5 loại `rest_short/rest_long/sse_short/sse_long/ws_short`.
- `app/observability.py`: cumulative CPU gateway/reaped children (FFmpeg) và lifetime RSS high-water.
- `tests/unit/test_benchmark.py`: thống kê, interval histogram, HTTP/HTTPS/SSE/WS/WSS, empty/truncated/error SSE,
  không ghi payload/secret vào report, CLI validation.
- `tests/unit/test_capacity_hardening.py`: simulated allocator OOM, recovery readiness barrier, hung probes,
  fresh-instance warm-up barrier và bounded mixed soak trả hết audio/spool/lane/job budgets.
- `tests/scheduler/test_scheduler.py`: drain, deadlines, failed recovery, hung warm-up, serialization/cancellation;
  `tests/audio/`: temp cleanup, spool/decoded bounds; `tests/unit/test_app.py`: auth/log metadata.
- `deploy/nginx.conf.example`: TLS/upgrade/SSE/no-replay/timeouts; là template, **chưa là proxy release evidence**.

Hardening phát hiện Python `base64.b64decode(validate=True)` vẫn nhận excess padding (`QUJD=`), làm test
bad-append chờ error mãi. Realtime nay kiểm tra canonical padded base64 trong cùng offload; packet lỗi
không thay buffer. Đây không thay đổi schema/event contract.

### Điều kiện đóng gate (operator phải điền bằng chứng)

1. GPU dành riêng, không tiến trình khác; ghi GPU/driver/CPU/RAM, CUDA extra, image digest, model manifest SHA,
   commit, config không chứa keys, fixture SHA, proxy config và protocol mix. Ghi phiên bản Python/torch/runtime.
2. Cold startup từ process mới (ghi thời điểm start và engine.loaded), warm ramp cho từng workload và mixed;
   ít nhất 3 lần lặp mỗi mức tải ứng viên, đủ samples cho p95 (≥100 thành công mỗi nhóm).
3. Chốt workload, SLO p95 và error budget **trước** khi xét kết quả. Ghi mức tải cao nhất đã thử, mức tải
   cao nhất đạt SLO, mức được chọn sau headroom và trường hợp overload. Không gọi số client closed-loop là
   sessions microphone liên tục hay arrival-rate capacity; nếu cần SLO arrival-rate phải thêm open-loop generator.
4. Soak ≥1 giờ tại mức đã chọn, qua proxy TLS thực; không tăng occupancy sau settle; RSS/VRAM có headroom;
   report không chạm request cap. Soak CPU vài giây chỉ test cleanup, không thay gate này.
5. Cùng deployment: SIGTERM drain/restart, process kill/restart, allocator OOM/recovery trên GPU, hung-worker
   watchdog; readiness chặn traffic cho tới warm-up xong. Không gây OOM trên host GPU chia sẻ.
6. Áp limits đo được bằng deployment env, chạy lại container GPU + SDK + benchmark regression. Ghi bảng
   quyết định và rollback digest; chỉ lúc đó đánh dấu toàn bộ M7 done.

## 2. Cách chạy lặp lại

Stage và verify fixtures/assets theo AGENTS.md. Chọn **GPU trống** bằng `nvidia-smi`; không kill tiến trình khác.
Chạy server một worker theo command chuẩn, lưu log metadata vào nơi quyền 0600; dùng port 18000 hoặc port trống.
`QUAN_GPU`/`QUAN_TEST_*` không được có trong env server. Không đưa keys vào argv/report.

```bash
# OPENAI_API_KEY đã được inject từ secret store, không ghi literal vào shell history.
uv run --extra cu128 scripts/benchmark.py --engine gpu --hardware 'RTX5090-exclusive-cu128' \
  --base-url http://127.0.0.1:18000 --clients 1,2,4,8,16,32 \
  --requests-per-client 100 --workload mixed --phase warm \
  --server-pid <host-server-pid> --out docs/baseline/m7-mixed-warm.json

# Chạy lại từng workload để không che tail latency bằng aggregate mixed.
# --workload short | long | sse | ws; fixture long mặc định fixtures-large/vi_long.wav.
# WS dùng tests/fixtures/vi_a.wav; --ws-pace realtime (20 ms packets) là default.
# Burst chỉ diagnostic, không được gọi là realtime capacity.

# Cold: process mới, không warm-up client; chỉ request đầu là cold-serving.
# Engine vẫn warm-up trước /ready. readiness_wait_s KHÔNG phải thời gian process startup.
uv run --extra cu128 scripts/benchmark.py --engine gpu --hardware 'RTX5090-exclusive-cu128' \
  --phase cold --clients 1 --requests-per-client 1 --workload short \
  --out docs/baseline/m7-cold.json

# Soak tại mức đã được chọn từ ramp; qua proxy, CA được verify cho cả HTTPS/WSS.
uv run --extra cu128 scripts/benchmark.py --engine gpu --hardware 'RTX5090-exclusive-cu128-via-proxy' \
  --base-url https://transcribe.example.com --ca /path/to/ca.pem \
  --clients <measured-client-limit> --duration 3600 --workload mixed \
  --server-pid <host-server-pid> --out docs/baseline/m7-soak.json
```

`--short`/`--long` nhận WAV fixtures do operator chọn; chỉ hash/duration xuất ra report. Không dùng audio nhạy cảm.
Không có LM mode trong server (greedy only); report ghi LM off, không giả kết quả LM on.

Exit 1 nếu có request lỗi (kể cả overload dự kiến) hoặc chưa đủ thời gian soak do cap; exit 0 **không** chứng nhận
production gate/SLO. Report ghi từng stage ngay sau hoàn tất, không lưu response text/audio/API key/base URL.
Hardware label là operator metadata, không nhập secrets vào đó. Mỗi stage giới hạn 256 client, tối đa 10.000
request mỗi client (kể cả soak); `duration_target_met=false` phải chạy lại với workload/thời lượng phù hợp.
Không chạy benchmark cùng traffic khác: server histogram/CPU là process-wide, không có request labels.
Không restart server giữa stage; phát hiện counter giảm thì hủy stage. Restart có counter mới lớn hơn counter cũ
không thể phát hiện chỉ bằng counter: operator phải đối chiếu lifecycle log.

## 3. Định nghĩa số đo và giới hạn của phép đo

| Field | Ý nghĩa |
|---|---|
| REST `final_s` | Từ trước serialize multipart/upload tới đọc xong JSON; gồm upload/decode/queue/infer/download, không phải GPU-only latency |
| SSE `first_delta_s` | Cùng mốc upload-start, chỉ khi có `transcript.text.delta`; không lấy header/keepalive/done thay delta; empty transcript → null |
| SSE `final_s` | Upload-start → nhận `transcript.text.done`; kiểm tra EOF và joined deltas == done |
| WS `commit_to_first_delta_s` / `commit_to_final_s` | Từ ngay trước send commit tới delta/completed; không gồm paced audio append |
| WS `final_s` | Bao gồm handshake (nếu socket mới), append và commit; RTF WS theo field này, không trộn với RTF REST khi chốt SLO |
| `rtf` | final_s / audio_s của request **thành công**, nearest-rank p50/p95; lỗi có error/overload rate riêng |
| `successful_audio_s_per_wall_s` | Tổng audio thành công / stage wall; client-side wall gồm setup/teardown/sampling, không phải GPU throughput thuần |
| `server.queue_wait` | Submit → bắt đầu inference, histogram delta giữa hai lần scrape, gộp lane live/file; không gồm upload/decode/capacity admission wait |
| `server.decode_wall` | Decode wall, gồm chờ decode slot, không gọi là decode CPU |
| `children_cpu_seconds_total` | Delta CPU của children đã reap (FFmpeg trong server chuẩn), toàn stage; có thể gồm subprocess khác; không phải p95 CPU từng request |
| `process_cpu_seconds_total` | Delta CPU gateway process, không gồm GPU execution và children |
| RSS / VRAM sampled peak | Mẫu mỗi ~0.5s, có thể bỏ lỡ spike; null nếu không có local PID/quyền/nvidia-smi; VRAM theo PID, không lấy tổng GPU |

Queue/decode/infer p50/p95 là **upper bucket bounds**, không phải quantile chính xác/interpolation. Không có
observations hoặc quantile vượt bucket 120s → null; xem count/overflow_count để phân biệt. Cần histogram dày hơn
nếu SLO nằm trong một bucket. `process_max_rss_bytes` ở `/metrics` là peak cả lifetime, không peak từng stage.
PID container phải là **host PID** cho sampler. GPU peak khi startup phải đo riêng; benchmark chờ ready nên không
bao phủ loader peak. Client chuẩn bị/resample PCM trước stage, không cộng CPU preprocessing client vào server.

## 4. Limits và bảng quyết định (chưa có số production)

Default config hiện là **finite safety budgets**, không phải capacity đã chứng minh:

| Resource | Default hiện tại | Quy tắc chọn sau đo |
|---|---:|---|
| Lanes / sessions | 32 / 16 | Không tăng theo GPU util đơn lẻ; WS giữ lane suốt session; chừa lane cho REST |
| Queue jobs / mỗi lane | 64 / 2 | Chốt queue delay, không tăng queue để giấu overload |
| Queue audio / decoded audio | 900s / 3600s | Float32 mono 16k: 64.000 byte/s, chưa tính bản sao/frontend/PCM24 |
| Upload / spool | 25 MiB / 200 MiB | Cả multipart body; cần dung lượng thêm cho caches/temp và filesystem headroom |
| Decode concurrency | 2 | Từ CPU/FFmpeg saturation và mixed tail latency |
| Chunk / turn / file | 30s / 120s / 1800s | Theo VRAM peak, request latency và ingress bounds |
| Queue / request / infer hang | 30s / 600s / 120s | Từ p95/p99 và watchdog, không tùy tiện nới hang để che treo |
| Drain / send | 30s / 10s | Client chậm bị đóng; không giải phóng running-job ownership sớm |

Maximum tested **GPU load: chưa có**. Production client/session limit: **chưa chốt**. Chưa sửa defaults.
Bảng kết quả release cần: workload + hardware + config hash, highest tested clients, highest SLO-pass clients,
selected limit/headroom, p95 queue/first/final, error/429 rate, RSS/VRAM peak, soak report và restart evidence.
Replica/GPU + sticky WS, REST microbatch và KenLM vẫn là optimization riêng; không triển khai trước capacity gate.

## 5. Runbook vận hành

### TLS/proxy và disk

Dùng `deploy/nginx.conf.example`, `nginx -t` rồi kiểm thử SSE flush, 101 WS upgrade, 401 trước accept,
25 MiB body limit, client disconnect và long request qua proxy. Không tắt verify cert; benchmark `--ca` cho CA riêng.
Timeout proxy phải lớn hơn request deadline; idle WS có protocol pings nhưng vẫn chịu session/idle limits gateway.
Không retry POST/WS tự động; idempotency/replay không được hỗ trợ. Client reconnect là session mới, không resume.
Template không tự triển khai/cert-rotate và chưa thay kiểm thử ingress thật. Nếu proxy chặn `/metrics`, chạy generator
trên management host được allowlist hoặc mở ACL quản trị phù hợp (vẫn cần Bearer); không expose metrics public.

Spool ở private 0700 subdir, file 0600; không dùng path client. Compose `/tmp` tmpfs 512 MiB với spool budget
200 MiB: tính cả caches và RAM charge của tmpfs vào container memory. Alert disk/tmpfs headroom, spool occupancy,
active decodes/decoded audio, queue và worker_hung; không chỉ VRAM. Verify temp cleanup sau cancel và ENOSPC
trong isolated test/container, không lấp disk host chung. Rotate/limit container logs ở deployment logging driver;
không enable access/debug body logs ở proxy. Nginx error logs có thể chứa URL: không truyền secret qua URL và
restrict quyền/truy cập/retention. Gateway/proxy metadata không được chứa transcript/audio/filenames/key/token.

### Drain, hung worker và restart

1. Bỏ replica khỏi LB readiness/routing **trước** SIGTERM; ngừng upgrade socket mới. Không có admin drain HTTP API.
2. Báo client đóng/commit trước maintenance. WS lâu sống không di chuyển sang replica khác; deploy rolling cần
   ít nhất replica khác đã warm/ready. Không replay turn chưa rõ completed sau network failure.
3. SIGTERM server/container: Uvicorn graceful 30s, lifespan scheduler drain tối đa 30s, worker stop thêm 5s.
   Compose stop_grace_period 70s đã chừa khoảng này. Sau deadline có thể mất request; client xử lý lỗi rõ ràng.
4. `/ready` không được route trước load/warm-up, khi recovering/hung hoặc scheduler closing. `/health` có thể
   vẫn 200 khi recover đang warm-up; hung/load/recovery failure → 503, external watchdog restart process.
   Docker `restart: unless-stopped` không restart chỉ vì health status unhealthy: cần orchestrator/watchdog.
5. Running inference không cancel được an toàn; không tạo worker thứ hai trên model cũ. Nếu treo không kết thúc,
   terminate process sau grace, SIGKILL từ supervisor nếu cần; GPU context/budget thu hồi ở process exit.
6. Worker có thể tự warm-up sau exception; recovery failure giữ unready/unhealthy. CUDA OOM lặp lại: drain/restart,
   hạ limits về profile đã biết tốt, không retry vô hạn. Fake OOM test không chứng minh CUDA context còn lành.
7. Restart: `/health` rồi `/ready`, SDK smoke và fixture parity trước trả traffic; không xem process running là ready.

### Secrets và rollback

Keys đọc một lần lúc startup, **không hot reload file**. Rotate hai bước: deploy file `old + new` bằng atomic replace,
rolling restart/wait ready, chuyển clients sang new, deploy file chỉ new rồi restart lại; verify old=401/new=200.
Không dump env/config chứa keys, không truyền key bằng CLI; secret file 0400 theo UID container. HF_TOKEN chỉ cho
staging, không đưa vào serving container. CPU tests kiểm chứng envelope/log metadata, không thay rotation drill thật.

Rollback: lưu image digest + CUDA profile + manifest/config (không secrets) của bản tốt; drain như trên, mount lại
model ro tương ứng, restore limits profile tốt và deploy digest cũ. Không sửa model pin tại chỗ và không kéo Hub main.
Chờ ready, chạy smoke parity/SDK, rồi mở LB. Rollback không phục hồi WS session; thông báo client reconnect.

## 6. Bằng chứng hiện có

`docs/baseline/m7-cpu-diagnostic.json`: fake engine delay 3 ms/chunk, fixtures thật 8.88s/106.62s,
WS burst, mỗi stage tối thiểu 3s: clients 1/4/16 → 76/223/235 requests, đều thành công; RSS/VRAM null vì không
bật PID sampler. Đây chỉ kiểm tra harness/mixed transports, **không có inference/GPU capacity**.

CPU gate:

```bash
CUDA_VISIBLE_DEVICES='' uv run --extra cu128 --offline pytest tests/unit tests/scheduler tests/audio tests/contract -q
# 429 passed in 38.30s
```

Bao gồm Python + JS SDK server contracts. Các gate GPU thật, soak ≥1h, proxy deployment, real process/container
restart và limit decision còn mở cho tới khi có GPU riêng. Không coi kết quả fake 16 clients là production limit.
