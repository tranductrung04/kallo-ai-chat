import asyncio
import json
import secrets
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, settings
from .llm import LLMClient, LLMError, ProviderConfig
from .models import ChatRequest, ErrorBody
from .pipeline import build_prompt, clamp_max_tokens, sanitize_response


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: object) -> bytes:
        return json.dumps(content, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=str).encode("utf-8")


app = FastAPI(title="Kallo AI Chat", version="0.2.0", default_response_class=UTF8JSONResponse)
config: Settings = settings()
providers = [
    ProviderConfig("primary", config.llm_base_url, config.llm_api_key, config.llm_model, config.llm_reasoning_effort, config.llm_extra_body),
]
if config.fallback_base_url and config.fallback_api_key and config.fallback_model:
    providers.append(ProviderConfig("fallback", config.fallback_base_url, config.fallback_api_key, config.fallback_model, config.llm_reasoning_effort, config.llm_extra_body))
llm = LLMClient(providers, connect_timeout=config.llm_connect_timeout, first_token_timeout=config.llm_first_token_timeout, timeout=config.llm_timeout, max_retries=config.llm_max_retries, hard_max_tokens=config.hard_max_tokens)
generation_lock = asyncio.Semaphore(config.max_concurrent_generations)


def error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": ErrorBody(code=code, message=message).model_dump()}, headers=headers)


def authorized(api_key: str | None) -> bool:
    return bool(config.api_key) and bool(api_key) and secrets.compare_digest(api_key, config.api_key)


def llm_error_response(exc: LLMError) -> JSONResponse:
    if exc.status_code == 401 or exc.status_code == 403:
        return error(502, "llm_error", "Dịch vụ AI chưa sẵn sàng.")
    if exc.status_code == 429:
        return error(503, "llm_rate_limited", "AI đang hết lượt, thử lại sau.", {"Retry-After": "60"})
    if "timed out" in str(exc).lower():
        return error(504, "llm_timeout", "AI phản hồi quá lâu.")
    return error(503, "llm_unavailable", "AI tạm thời không khả dụng.")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    if not await llm.health():
        return error(503, "model_loading", "LLM chưa sẵn sàng.")
    return JSONResponse({"status": "ready"})


@app.post("/v1/chat")
async def chat(payload: ChatRequest, x_internal_api_key: str | None = Header(default=None)):
    if not authorized(x_internal_api_key):
        return error(401, "unauthorized", "API key không hợp lệ.")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=config.max_queue_wait)
    except TimeoutError:
        return error(503, "busy", "AI đang bận, thử lại sau.", {"Retry-After": str(int(config.max_queue_wait))})
    try:
        system, messages = build_prompt(payload)
        if payload.stream:
            return StreamingResponse(stream_response(payload, system, messages), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        text = sanitize_response(await llm.complete(messages, system, clamp_max_tokens(payload.max_tokens, config.max_tokens, config.hard_max_tokens)), payload.system_context.allergies)
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
    except LLMError as exc:
        return llm_error_response(exc)
    finally:
        if not payload.stream:
            generation_lock.release()


async def stream_response(payload: ChatRequest, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
    try:
        buffer = ""
        async for delta in llm.stream(messages, system, clamp_max_tokens(payload.max_tokens, config.max_tokens, config.hard_max_tokens)):
            buffer += delta
        text = sanitize_response(buffer, payload.system_context.allergies)
        yield f"data: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except LLMError as exc:
        code = "llm_rate_limited" if exc.status_code == 429 else "llm_error"
        yield f"event: error\ndata: {json.dumps({'code': code, 'message': 'AI không trả lời được.'}, ensure_ascii=False)}\n\n"
    finally:
        generation_lock.release()
