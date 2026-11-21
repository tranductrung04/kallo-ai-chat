import asyncio

import httpx

from app.llm import LLMClient, ProviderConfig


def test_payload_uses_bearer_and_omits_llamacpp_fields(monkeypatch):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        seen["json"] = request.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Xin chào"}}]})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(transport=transport, **kwargs))
    client = LLMClient([ProviderConfig("test", "https://llm.test/v1", "secret", "model")])

    completion = asyncio.run(client.complete([], "system", 20))
    assert completion.content == "Xin chào"
    assert completion.finish_reason == "stop"
    assert seen["authorization"] == "Bearer secret"
    assert b"cache_prompt" not in seen["json"]


def test_fallback_after_primary_429(monkeypatch):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "fallback"}}]})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(transport=transport, **kwargs))
    client = LLMClient(
        [
            ProviderConfig("primary", "https://primary.test/v1", "one", "model"),
            ProviderConfig("fallback", "https://fallback.test/v1", "two", "model"),
        ],
        max_retries=0,
    )

    assert asyncio.run(client.complete([], "system", 20)).content == "fallback"
    assert calls == ["primary.test", "fallback.test"]
