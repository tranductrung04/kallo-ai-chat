import asyncio
import json
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, settings
from .llm import LLMClient, LLMError, ProviderConfig
from .models import ChatRequest, ErrorBody
from .pipeline import build_prompt, clamp_max_tokens, sanitize_response


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: object) -> bytes:
        return json.dumps(content, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=str).encode("utf-8")


config: Settings = settings()
providers = [
    ProviderConfig("primary", config.llm_base_url, config.llm_api_key, config.llm_model, config.llm_reasoning_effort, config.llm_extra_body),
]
if config.fallback_base_url and config.fallback_api_key and config.fallback_model:
    providers.append(ProviderConfig("fallback", config.fallback_base_url, config.fallback_api_key, config.fallback_model, config.fallback_reasoning_effort, config.fallback_extra_body))
llm = LLMClient(providers, connect_timeout=config.llm_connect_timeout, first_token_timeout=config.llm_first_token_timeout, timeout=config.llm_timeout, max_retries=config.llm_max_retries, hard_max_tokens=config.hard_max_tokens)
generation_lock = asyncio.Semaphore(config.max_concurrent_generations)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await llm.close()


app = FastAPI(title="Kallo AI Chat", version="0.2.0", lifespan=lifespan, default_response_class=UTF8JSONResponse)


def error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": ErrorBody(code=code, message=message).model_dump()}, headers=headers)


def authorized(api_key: str | None) -> bool:
    if not config.api_key or not api_key:
        return False
    try:
        return secrets.compare_digest(api_key.encode("utf-8"), config.api_key.encode("utf-8"))
    except UnicodeEncodeError:
        return False


@app.middleware("http")
async def require_internal_api_key(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/v1/chat" and not authorized(request.headers.get("X-Internal-Api-Key")):
        return error(401, "unauthorized", "API key không hợp lệ.")
    return await call_next(request)


def llm_error_response(exc: LLMError) -> JSONResponse:
    if exc.blocked:
        return JSONResponse({"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mình chưa thể tạo câu trả lời an toàn lúc này. Bạn thử hỏi lại theo hướng dinh dưỡng phù hợp nhé."}, "finish_reason": "content_filter"}]})
    if exc.status_code == 401 or exc.status_code == 403:
        return error(502, "llm_error", "Dịch vụ AI chưa sẵn sàng.")
    if exc.status_code == 429:
        return error(503, "llm_rate_limited", "AI đang hết lượt, thử lại sau.", {"Retry-After": "3"})
    if exc.status_code is not None and 400 <= exc.status_code < 500:
        return error(502, "llm_error", "Dịch vụ AI từ chối yêu cầu.")
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
async def chat(request: Request, payload: ChatRequest):
    system, messages = build_prompt(payload, config.history_turns)
    if payload.effective_stream:
        return StreamingResponse(
            stream_response(payload, system, messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    acquired = False
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=config.max_queue_wait)
        acquired = True
        completion = await llm.complete(
            messages,
            system,
            clamp_max_tokens(payload.effective_max_tokens, config.max_tokens, config.hard_max_tokens),
        )
        text = sanitize_response(completion.content, payload.effective_system_context.allergies)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "message": {"role": "assistant", "content": text},
            "suggestedFoods": [],
            "usage": {"promptTokens": None, "completionTokens": None},
            "modelVersion": config.llm_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": completion.finish_reason,
            }],
        }
    except asyncio.TimeoutError:
        return error(503, "busy", "Hệ thống đang bận, thử lại sau.", {"Retry-After": "5"})
    except LLMError as exc:
        return llm_error_response(exc)
    finally:
        if acquired:
            generation_lock.release()


async def stream_response(payload: ChatRequest, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
    acquired = False
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=config.max_queue_wait)
        acquired = True
        yield ": ping\n\n"
        buffer = ""
        pending = ""
        finish_reason = "stop"
        async for chunk in llm.stream(messages, system, clamp_max_tokens(payload.effective_max_tokens, config.max_tokens, config.hard_max_tokens)):
            if chunk.content:
                buffer += chunk.content
                pending += chunk.content
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            while "\n" in pending:
                sentence, pending = pending.split("\n", 1)
                yield f"event: delta\ndata: {json.dumps({'content': sentence + chr(10), 'text': sentence + chr(10)}, ensure_ascii=False)}\n\n"
        if pending:
            yield f"event: delta\ndata: {json.dumps({'content': pending, 'text': pending}, ensure_ascii=False)}\n\n"
        text = sanitize_response(buffer, payload.effective_system_context.allergies)
        if text != buffer.strip():
            yield f"event: replace\ndata: {json.dumps({'content': text, 'text': text}, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({'content': text, 'text': text, 'finish_reason': finish_reason, 'suggestedFoods': [], 'usage': {}, 'modelVersion': config.llm_model}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        yield f"event: error\ndata: {json.dumps({'code': 'busy', 'message': 'Hệ thống đang bận, thử lại sau.'}, ensure_ascii=False)}\n\n"
    except LLMError as exc:
        if exc.blocked:
            text = "Mình chưa thể tạo câu trả lời an toàn lúc này. Bạn thử hỏi lại theo hướng dinh dưỡng phù hợp nhé."
            yield f"event: done\ndata: {json.dumps({'content': text, 'finish_reason': 'content_filter'}, ensure_ascii=False)}\n\n"
        else:
            if "timed out" in str(exc).lower():
                code = "llm_timeout"
            elif exc.status_code == 429:
                code = "llm_rate_limited"
            else:
                code = "llm_error"
            yield f"event: error\ndata: {json.dumps({'code': code, 'message': 'AI không trả lời được.'}, ensure_ascii=False)}\n\n"
    finally:
        if acquired:
            generation_lock.release()
