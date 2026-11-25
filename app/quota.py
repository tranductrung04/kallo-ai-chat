from collections import deque
from datetime import date, datetime, timezone
from threading import Lock


class QuotaLimiter:
    def __init__(self, rpm: int, tpm: int, daily_requests: int) -> None:
        self.rpm = max(0, rpm)
        self.tpm = max(0, tpm)
        self.daily_requests = max(0, daily_requests)
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._day: date = date.today()
        self._daily_count = 0
        self._lock = Lock()

    def try_acquire(self, estimated_tokens: int) -> float | None:
        now = datetime.now(timezone.utc).timestamp()
        estimated_tokens = max(1, estimated_tokens)
        with self._lock:
            self._reset_day()
            self._discard_old(now)
            if self.daily_requests and self._daily_count >= self.daily_requests:
                return 86400 - (now % 86400)
            if self.rpm and len(self._requests) >= self.rpm:
                return max(0.1, 60 - (now - self._requests[0]))
            used_tokens = sum(tokens for _, tokens in self._tokens)
            if self.tpm and used_tokens + estimated_tokens > self.tpm:
                return max(0.1, 60 - (now - self._tokens[0][0]))
            self._requests.append(now)
            self._tokens.append((now, estimated_tokens))
            self._daily_count += 1
            return None

    def _reset_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._daily_count = 0

    def _discard_old(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= 60:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0][0] >= 60:
            self._tokens.popleft()


def estimate_tokens(system: str, messages: list[dict[str, object]], max_tokens: int) -> int:
    text = system + " " + " ".join(str(message.get("content", "")) for message in messages)
    return max(1, len(text) // 4) + max(1, max_tokens)
