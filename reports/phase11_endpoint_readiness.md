# Phase 11: Final Sequential Endpoint Readiness

**Date:** 2026-04-07  
**Status:** READY FOR COMPETITION

## Overview

This document defines the readiness checklist for the final sequential competition endpoint. The actual competition endpoint URL and behavior may differ from our mock server—this checklist ensures we can adapt quickly.

## Sequential Protocol Flow

```
1. POST /auth/             → { "token": "..." }
2. POST /session/open/     → { "session_id", "session_name", ... }
3. GET  /session/next/     → { "frame_url", "image_url", ... } or 204
4. GET  /media/...         → image bytes
5. POST /session/prediction/ → 201 Created
6. (repeat 3-5 until 204)
7. POST /session/close/    → 200 OK
```

## Readiness Checklist

### 1. Login / Authentication

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| Endpoint | `POST /auth/` | `SequentialProtocolSettings.auth_path` | ✅ Configurable |
| Payload | `username`, `password` (form-encoded) | `client.post_form()` | ✅ Supported |
| Response | `{ "token": "..." }` | `login()` extracts token | ✅ Ready |
| Token usage | `Authorization: Token {token}` | `_auth_headers()` | ✅ Ready |
| Failure handling | Non-200 raises `ProtocolError` | ✅ Implemented | ✅ Ready |

### 2. Session Open

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| Endpoint | `POST /session/open/` | `path_overrides["open_session"]` | ✅ Configurable |
| Response fields | `session_id`, `session_name` | Stored in adapter | ✅ Ready |
| Optional fields | `object_limit`, `reference_manifest` | Logged in diagnostics | ✅ Ready |
| Auto-login | If no token, calls `login()` first | ✅ Implemented | ✅ Ready |

### 3. Frame Fetch

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| Endpoint | `GET /session/next/` | `path_overrides["next_frame"]` | ✅ Configurable |
| Success response | 200 + JSON with frame data | Parsed to `FrameEnvelope` | ✅ Ready |
| End-of-session | 204 No Content | Returns `None` | ✅ Ready |
| Retry on | 408, 429, 500, 502, 503, 504 | `next_frame_retryable_statuses` | ✅ Configurable |
| First frame wait | Extended timeout for warmup | `first_frame_timeout_s` | ✅ Ready |

**Required frame JSON fields:**
```json
{
  "frame_url": "http://.../frames/123/",
  "image_url": "/media/frames/123.jpg",
  "video_name": "session_video",
  "translation_x": 0.0,
  "translation_y": 0.0,
  "translation_z": 5.0,
  "health_status": "1"
}
```

**Optional frame metadata:**
- `session_id`, `frame_id`, `camera_mode`, `deadline_ms`

### 4. Image Download

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| URL resolution | Relative → `{base_url}media/{path}` | `resolve_image_url()` | ✅ Ready |
| Absolute URL | Passed through directly | ✅ Handled | ✅ Ready |
| Timeout | Configurable | `image_timeout_s` | ✅ Ready |

### 5. Prediction Submission

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| Endpoint | `POST /session/prediction/` | `path_overrides["prediction"]` | ✅ Configurable |
| Success response | 201 Created | Validated | ✅ Ready |
| Retry on | 408, 429, 500, 502, 503, 504 | `prediction_retryable_statuses` | ✅ Configurable |

**Prediction payload schema:**
```json
{
  "session_id": "abc123",
  "frame_id": "frame_001",
  "frame": "http://.../frames/123/",
  "detected_objects": [
    {
      "cls": "http://.../classes/1/",
      "landing_status": "0",
      "top_left_x": "100",
      "top_left_y": "200",
      "bottom_right_x": "300",
      "bottom_right_y": "400"
    }
  ],
  "detected_translations": [
    {
      "translation_x": "0.5",
      "translation_y": "-0.3",
      "translation_z": "5.2"
    }
  ]
}
```

### 6. Session Close

| Item | Expected | Adapter Support | Status |
|------|----------|-----------------|--------|
| Endpoint | `POST /session/close/` | `path_overrides["close_session"]` | ✅ Configurable |
| Payload | `{ "session_id": "..." }` | ✅ Sent | ✅ Ready |
| Cleanup | Clears adapter state | ✅ Implemented | ✅ Ready |

## Retry / Failure Behavior

### Transient Failure Handling

| Scenario | Behavior | Configurable |
|----------|----------|--------------|
| Network timeout | Retry with backoff | `retry_policy.max_retries`, `retry_policy.backoff_s` |
| HTTP 408/429/5xx | Retry with backoff | `*_retryable_statuses` lists |
| Retry exhausted | Raise `ProtocolError` | - |
| Login failure | Raise `ProtocolError` immediately | - |
| Invalid JSON | Raise `ProtocolError` | - |

### Sequential Ordering Guarantee

**CRITICAL:** The adapter **never** sends a prediction for frame N+1 before completing frame N.

- `current_frame` is cleared after each successful `send_wire_prediction()`
- `fetch_next_frame()` overwrites `current_frame` only after server responds
- No parallel frame processing allowed

## Configuration Flexibility

All paths can be overridden via `SequentialProtocolSettings.path_overrides`:

```python
SequentialProtocolSettings(
    base_url="https://competition.teknofest.org/api/",
    username="team_xyz",
    password="secret",
    path_overrides={
        "auth": "v2/authenticate/",
        "open_session": "v2/session/start/",
        "next_frame": "v2/session/frame/",
        "prediction": "v2/session/submit/",
        "close_session": "v2/session/end/",
    },
    timeout_policy={
        "request_timeout_s": 10.0,
        "first_frame_timeout_s": 30.0,
        "image_timeout_s": 15.0,
    },
)
```

## Pre-Competition Validation Commands

```bash
# 1. Smoke test with mock server
python tools/run_runtime_smoke.py --mode sequential --frames 5

# 2. Check adapter settings without running
python -c "from src.config.settings import SequentialProtocolSettings; print(SequentialProtocolSettings.__doc__)"

# 3. Validate schema compliance
python -m pytest tests/test_sequential_interface.py tests/test_sequential_warmup_retry.py -v
```

## Competition Day Quick Adaptation

If the final endpoint differs from mock:

1. **Path changes:** Update `path_overrides` in config
2. **Timeout changes:** Update `timeout_policy`
3. **New required fields:** Check `build_wire_prediction()` in `final_sequential_adapter.py`
4. **Auth method change:** May need adapter modification (escalate immediately)

## Known Assumptions

These assumptions must be validated when final endpoint is published:

| Assumption | Fallback if wrong |
|------------|-------------------|
| Token auth via `Authorization: Token X` | Update `_auth_headers()` |
| JSON content type | Update `client.post_json()` |
| 204 signals end-of-session | Check for alternative signals |
| String values in prediction coords | May need numeric conversion |
| Class URL format `{base}/classes/{id}/` | Update `build_wire_prediction()` |

---

*This checklist ensures the runtime is ready for the unpublished final sequential endpoint.*
