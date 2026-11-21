from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio
import json
import random
import time
from typing import Any

import httpx


class LLMError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    extra_body: dict[str, object] | None = None


@dataclass
class _ProviderState:
    failures: int = 0
    open_until: float = 0.0


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None


class LLMClient:
    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        connect_timeout: float = 5,
        first_token_timeout: float = 30,
        timeout: float = 90,
        max_retries: int = 2,
        hard_max_tokens: int = 384,
    ):
        self.providers = providers
        self.connect_timeout = connect_timeout
        self.first_token_timeout = first_token_timeout
        self.timeout = timeout
        self.max_retries = max_retries
        self.hard_max_tokens = hard_max_tokens
        self._states = {provider.name: _ProviderState() for provider in providers}

    def _payload(self, provider: ProviderConfig, messages: list[dict[str, Any]], system: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": min(max_tokens, self.hard_max_tokens),
            "temperature": 0.4,
            "stream": stream,
        }
        if provider.reasoning_effort:
            payload["reasoning_effort"] = provider.reasoning_effort
        if provider.extra_body:
            payload.update(provider.extra_body)
        return payload

    def _available(self, provider: ProviderConfig) -> bool:
        return time.monotonic() >= self._states[provider.name].open_until

    def _record_success(self, provider: ProviderConfig) -> None:
        state = self._states[provider.name]
        state.failures = 0
        state.open_until = 0

    def _record_failure(self, provider: ProviderConfig) -> None:
        state = self._states[provider.name]
        state.failures += 1
        if state.failures >= 3:
            state.open_until = time.monotonic() + 60

    async def health(self) -> bool:
        return bool(self.providers and all(provider.api_key and provider.model for provider in self.providers[:1]))

    async def complete(self, messages: list[dict[str, Any]], system: str, max_tokens: int) -> str:
        last_error: LLMError | None = None
        for provider in self.providers:
            if not self._available(provider):
                continue
            for attempt in range(self.max_retries + 1):
                response: httpx.Response | None = None
                try:
                    timeout = httpx.Timeout(self.timeout, connect=self.connect_timeout)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(
                            f"{provider.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {provider.api_key}"},
                            json=self._payload(provider, messages, system, max_tokens, False),
                        )
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise LLMError("provider temporarily unavailable", status_code=response.status_code, retryable=True)
                    if response.status_code in {401, 403}:
                        raise LLMError("provider authentication failed", status_code=response.status_code)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise LLMError("empty LLM response")
                    self._record_success(provider)
                    return content
                except LLMError as exc:
                    last_error = exc
                    if not exc.retryable or attempt >= self.max_retries:
                        break
                    delay = _retry_after(response) if response is not None else None
                    await asyncio.sleep(delay if delay is not None else min(2 ** attempt, 8) + random.random())
                except httpx.TimeoutException:
                    last_error = LLMError("LLM request timed out", retryable=True)
                    if attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                    last_error = LLMError("LLM request failed", retryable=True)
                    if attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
            self._record_failure(provider)
        raise last_error or LLMError("no LLM provider available", retryable=True)

    async def stream(self, messages: list[dict[str, Any]], system: str, max_tokens: int) -> AsyncIterator[str]:
        last_error: LLMError | None = None
        for provider in self.providers:
            if not self._available(provider):
                continue
            try:
                timeout = httpx.Timeout(self.timeout, connect=self.connect_timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{provider.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {provider.api_key}"},
                        json=self._payload(provider, messages, system, max_tokens, True),
                    ) as response:
                        if response.status_code in {429, 500, 502, 503, 504}:
                            raise LLMError("provider temporarily unavailable", status_code=response.status_code, retryable=True)
                        if response.status_code in {401, 403}:
                            raise LLMError("provider authentication failed", status_code=response.status_code)
                        response.raise_for_status()
                        first = True
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            if not raw:
                                continue
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            choices = data.get("choices") or []
                            if not choices:
                                continue
                            delta = (choices[0].get("delta") or {}).get("content")
                            if delta:
                                first = False
                                yield delta
                        self._record_success(provider)
                        return
            except (LLMError, httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc if isinstance(exc, LLMError) else LLMError("LLM stream failed", retryable=True)
                self._record_failure(provider)
        raise last_error or LLMError("no LLM provider available", retryable=True)
