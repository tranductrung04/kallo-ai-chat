from app.main import chat
from app.models import ChatRequest
from app.quota import QuotaLimiter, estimate_tokens
from app.safety import safety_redirect


def _step_sync(coro):
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine yielded unexpectedly")


def _drain_async_iter(iterator):
    chunks = []
    while True:
        try:
            chunks.append(_step_sync(iterator.__anext__()))
        except StopAsyncIteration:
            return "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks)


def test_quota_limits_requests_and_estimates_tokens():
    limiter = QuotaLimiter(rpm=1, tpm=100, daily_requests=1)
    assert limiter.try_acquire(10) is None
    assert limiter.try_acquire(10) is not None
    assert estimate_tokens("abcd", [{"content": "efgh"}], 10) == 12


def test_safety_redirects_sensitive_requests_without_matching_normal_text():
    assert safety_redirect("Tôi muốn nhịn ăn để giảm cân")[1] == "safety_redirect"
    assert safety_redirect("Tôi muốn tự sát")[1] == "safety_redirect"
    assert safety_redirect("Tôi bị đau ngực")[1] == "safety_redirect"
    assert safety_redirect("Tôi muốn bỏ bữa để giảm cân")[1] == "safety_redirect"
    assert safety_redirect("Tôi muốn ăn cực đoan")[1] == "safety_redirect"
    assert safety_redirect("Đau ngực dữ dội")[1] == "safety_redirect"
    assert safety_redirect("Tôi nên ăn bao nhiêu protein?") is None


def test_safety_redirect_sse_uses_real_newlines():
    payload = ChatRequest.model_validate({
        "messages": [{"role": "user", "content": "Tôi bị đau ngực"}],
        "options": {"stream": True},
    })
    response = _step_sync(chat(None, payload))
    body = _drain_async_iter(response.body_iterator)
    assert "\\n" not in body
    assert "event: done\ndata: " in body
    assert "\n\ndata: [DONE]\n\n" in body
