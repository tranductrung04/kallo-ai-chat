from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio
import json
import random
import re
import time
from typing import Any

import httpx


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        blocked: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.blocked = blocked


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    extra_body: dict[str, object] | None = None


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    finish_reason: str


@dataclass(frozen=True)
class LLMChunk:
    content: str
    finish_reason: str | None = None


@dataclass
class _ProviderState:
    failures: int = 0
    open_until: float = 0.0


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return min(3.0, max(0.0, float(value))) if value else None
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
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=connect_timeout))

    async def close(self) -> None:
        await self._client.aclose()

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

    @staticmethod
    def _clean_content(content: object) -> str:
        if not isinstance(content, str):
            return ""
        return re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

    @staticmethod
    def _response_error(response: httpx.Response) -> LLMError | None:
        status = response.status_code
        if status in {401, 403}:
            return LLMError("provider authentication failed", status_code=status)
        if status == 429:
            return LLMError("provider rate limited", status_code=status)
        if status in {408, 409, 500, 502, 503, 504}:
            return LLMError("provider temporarily unavailable", status_code=status, retryable=True)
        if status >= 400:
            return LLMError("provider rejected request", status_code=status)
        return None

    @staticmethod
    def _backoff(attempt: int, response: httpx.Response | None = None) -> float:
        retry_after = _retry_after(response) if response is not None else None
        return retry_after if retry_after is not None else min(2 ** attempt, 8) + random.random()

    async def complete(self, messages: list[dict[str, Any]], system: str, max_tokens: int) -> LLMCompletion:
        last_error: LLMError | None = None
        deadline = time.monotonic() + self.timeout
        for provider in self.providers:
            if not self._available(provider):
                continue
            for attempt in range(self.max_retries + 1):
                response: httpx.Response | None = None
                if time.monotonic() >= deadline:
                    break
                try:
                    response = await self._client.post(
                        f"{provider.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {provider.api_key}"},
                        json=self._payload(provider, messages, system, max_tokens, False),
                    )
                    provider_error = self._response_error(response)
                    if provider_error:
                        raise provider_error
                    data = response.json()
                    choice = data["choices"][0]
                    finish_reason = choice.get("finish_reason") or "stop"
                    if finish_reason == "content_filter":
                        raise LLMError("provider content filter", blocked=True)
                    content = self._clean_content((choice.get("message") or {}).get("content"))
                    if not content:
                        raise LLMError("empty LLM response")
                    self._record_success(provider)
                    return LLMCompletion(content, finish_reason)
                except LLMError as exc:
                    last_error = exc
                    if not exc.retryable or attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(self._backoff(attempt, response), max(0.0, deadline - time.monotonic())))
                except httpx.TimeoutException:
                    last_error = LLMError("LLM request timed out", retryable=True)
                    if attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(self._backoff(attempt), max(0.0, deadline - time.monotonic())))
                except httpx.HTTPError:
                    last_error = LLMError("LLM request failed", retryable=True)
                    if attempt >= self.max_retries:
                        break
                    await asyncio.sleep(min(self._backoff(attempt), max(0.0, deadline - time.monotonic())))
                except (ValueError, KeyError, IndexError, TypeError):
                    last_error = LLMError("LLM response was invalid")
                    break
            if last_error is not None and not last_error.blocked:
                self._record_failure(provider)
            if last_error is not None and last_error.blocked:
                break
        raise last_error or LLMError("no LLM provider available", retryable=True)

    async def stream(self, messages: list[dict[str, Any]], system: str, max_tokens: int) -> AsyncIterator[LLMChunk]:
        last_error: LLMError | None = None
        deadline = time.monotonic() + self.timeout
        for provider in self.providers:
            if time.monotonic() >= deadline:
                break
            if not self._available(provider):
                continue
            emitted = False
            for attempt in range(self.max_retries + 1):
                if time.monotonic() >= deadline:
                    break
                try:
                    request = self._client.stream(
                        "POST",
                        f"{provider.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {provider.api_key}"},
                        json=self._payload(provider, messages, system, max_tokens, True),
                    )
                    async with request as response:
                        provider_error = self._response_error(response)
                        if provider_error:
                            raise provider_error
                        finish_reason = "stop"
                        lines = response.aiter_lines().__aiter__()
                        first_token = True
                        while True:
                            try:
                                line = await asyncio.wait_for(
                                    anext(lines),
                                    timeout=self.first_token_timeout if first_token else self.timeout,
                                )
                            except StopAsyncIteration:
                                break
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
                            choice = choices[0]
                            reason = choice.get("finish_reason")
                            if reason:
                                finish_reason = reason
                            if reason == "content_filter":
                                raise LLMError("provider content filter", blocked=True)
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                emitted = True
                                first_token = False
                                yield LLMChunk(delta)
                        if not emitted:
                            raise LLMError("empty LLM stream")
                        self._record_success(provider)
                        yield LLMChunk("", finish_reason)
                        return
                except asyncio.TimeoutError:
                    last_error = LLMError(
                        "LLM first token timed out" if not emitted else "LLM stream timed out",
                        retryable=not emitted,
                    )
                except LLMError as exc:
                    last_error = exc
                except httpx.TimeoutException:
                    last_error = LLMError("LLM stream timed out", retryable=not emitted)
                except httpx.HTTPError:
                    last_error = LLMError("LLM stream failed", retryable=not emitted)
                except (ValueError, KeyError, IndexError, TypeError):
                    last_error = LLMError("LLM stream response was invalid")
                if emitted or last_error.blocked or not last_error.retryable or attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(self._backoff(attempt), max(0.0, deadline - time.monotonic())))
            if last_error is not None and not last_error.blocked:
                self._record_failure(provider)
            if last_error is not None and (last_error.blocked or emitted):
                break
        raise last_error or LLMError("no LLM provider available", retryable=True)
