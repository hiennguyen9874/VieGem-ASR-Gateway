# Lỗi HTTP

Envelope (ví dụ `examples/error.json`):

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "language", "code": "unsupported_language"}}
```

| HTTP | `type` | `code` |
|---|---|---|
| 400 | `invalid_request_error` | `missing_required_parameter`, `unsupported_parameter`, `invalid_value` (kể cả multipart sai), `unsupported_language`, `unsupported_response_format`, `invalid_audio` (`param: "file"`: không decode được, định dạng không cho phép, decode quá giờ, sample không hữu hạn), `audio_too_long` |
| 401 | `authentication_error` | `invalid_api_key` (thiếu hoặc sai) |
| 404 | `invalid_request_error` | `model_not_found`; `not_found` (route không tồn tại, chỉ sau auth hợp lệ) |
| 405 | `invalid_request_error` | `method_not_allowed` |
| 413 | `invalid_request_error` | `request_too_large` (body vượt giới hạn, hoặc form field > 8 KiB) |
| 429 | `rate_limit_error` | `server_busy` (queue, decode slot, spool, budget audio đầy); kèm `Retry-After` khi có ước lượng |
| 500 | `server_error` | `inference_failed`, `alignment_failed` (`srt`/`vtt`), `internal_error` (không lộ chi tiết) |
| 503 | `server_error` | `model_not_ready` (`Retry-After: 5`), `shutting_down`, `request_timeout` |

- Mọi response có header `x-request-id: req_…` (Python SDK: `request_id`; JS SDK: `requestID`).
- Không bao giờ trả FastAPI 422 `{"detail": …}`. Message không chứa transcript, audio, token hay đường dẫn nội bộ.
- SDK ánh xạ status → class (`BadRequestError`, `AuthenticationError`, …). `examples/service_errors.json` được kiểm thử bằng cả hai SDK.
- API key: ≥ 16 ký tự ASCII printable, không khoảng trắng.

## Endpoint vận hành

Không thuộc profile OpenAI:

| Endpoint | Auth | Response |
|---|---|---|
| `GET /health` | Không | Liveness; 503 `unhealthy` khi cần restart process |
| `GET /ready` | Không | 200 `{"status": "ready"}` hoặc 503 |
| `GET /metrics` | Có | Prometheus text |
