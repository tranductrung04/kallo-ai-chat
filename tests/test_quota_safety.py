from app.quota import QuotaLimiter, estimate_tokens
from app.safety import safety_redirect


def test_quota_limits_requests_and_estimates_tokens():
    limiter = QuotaLimiter(rpm=1, tpm=100, daily_requests=1)
    assert limiter.try_acquire(10) is None
    assert limiter.try_acquire(10) is not None
    assert estimate_tokens("abcd", [{"content": "efgh"}], 10) == 12


def test_safety_redirects_sensitive_requests_without_matching_normal_text():
    assert safety_redirect("Tôi muốn nhịn ăn để giảm cân")[1] == "safety_redirect"
    assert safety_redirect("Tôi muốn tự sát")[1] == "safety_redirect"
    assert safety_redirect("Tôi nên ăn bao nhiêu protein?") is None
