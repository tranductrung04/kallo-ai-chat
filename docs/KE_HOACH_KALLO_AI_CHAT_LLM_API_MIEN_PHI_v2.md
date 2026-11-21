# Kế hoạch triển khai kallo-ai-chat với API LLM miễn phí (Gemini / Groq...) trên Hugging Face Space CPU free

> Phiên bản 2, cập nhật 21/11/2026. Thay thế bản "Qwen2.5-7B-Instruct Q4_K_M chạy trên HF Space CPU".
> Phạm vi: AI chat service cho module chat của `kallo-backend`. Space chỉ chạy FastAPI + truy xuất kiến thức; việc sinh văn bản do API LLM bên ngoài đảm nhiệm qua `openai_compat.py`.
> Trạng thái: kế hoạch mức demo / đồ án / beta nội bộ. Hạn mức, tên model và điều khoản của các gói free **thay đổi thường xuyên**; mọi con số trong tài liệu này là điểm xuất phát, phải đối chiếu lại trên trang chính thức và đo thật ở Giai đoạn 0.

---

## 0. Tóm tắt thay đổi so với bản trước

| Hạng mục | Bản cũ (tự host Qwen 7B) | Bản này (API miễn phí) |
|---|---|---|
| Nơi chạy LLM | llama-server trong cùng container | Nhà cung cấp bên ngoài (Gemini/Groq/...) |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | URL OpenAI-compatible của nhà cung cấp, kèm `LLM_API_KEY` |
| Nút thắt hiệu năng | CPU 2 vCPU (1-3 token/s) | Hạn mức free tier (RPM / TPM / RPD) và độ ổn định của nhà cung cấp |
| TTFT dự kiến | 20-45 giây | Vài giây (đo thật ở Giai đoạn 0) |
| Docker image | Có llama.cpp + model ~4,7GB | Chỉ Python + embedding nhỏ + index kiến thức |
| Cold start | Nạp model rất lâu | Chỉ nạp embedding và index |
| Rủi ro lớn nhất | Quá chậm, timeout | **Dữ liệu sức khỏe gửi ra ngoài**, hết quota, nhà cung cấp đổi chính sách |
| Tự host | Mặc định | Chuyển thành phương án dự phòng cuối (Phụ lục A) |

Phần giữ nguyên về nguyên tắc: pipeline (ngữ cảnh -> truy xuất -> FACTS -> prompt -> LLM -> hậu xử lý), FACTS do code tính, lọc dị ứng trên cả văn bản, `LLMClient` Protocol, hợp đồng `POST /v1/chat` với backend.

---

## 1. Mục tiêu và giới hạn

### 1.1 Mục tiêu
- Cung cấp `POST /v1/chat` (JSON hoặc SSE) đúng hợp đồng với backend.
- Số liệu dinh dưỡng do code tính (FACTS), model chỉ diễn giải.
- Không gợi ý món chứa chất dị ứng; không chẩn đoán; không đưa mức ăn cực đoan.
- Chi phí hạ tầng: 0 đồng (HF Space free + gói free của nhà cung cấp LLM).
- Đổi nhà cung cấp hoặc model chỉ bằng biến môi trường, không sửa pipeline.

### 1.2 Giới hạn của nền tảng (cần chấp nhận)
| Hạng mục | Thực tế | Hệ quả |
|---|---|---|
| HF Space free CPU | 2 vCPU, 16GB RAM, đĩa tạm thời | Chỉ đủ cho FastAPI + embedding nhỏ + index; RAM dư nhiều khi không còn LLM |
| Ngủ | Space tự ngủ khi không có traffic | Cold start (nhẹ hơn bản cũ), cần `/ready` và cơ chế đánh thức |
| Cổng | Chỉ mở một cổng (7860) | FastAPI mở ra ngoài; không còn llama-server nội bộ |
| Mạng ra ngoài | Space phải gọi được HTTPS tới nhà cung cấp | Kiểm tra ở Giai đoạn 0 |
| Hạn mức free tier | Giới hạn RPM / TPM / RPD theo từng model, theo project (không theo key). Gemini hiện không công bố bảng cố định, xem trực tiếp trong AI Studio; Groq xem trong console | Cần rate limiter phía mình, xử lý 429, ngân sách lượt/ngày |
| Dữ liệu | Gói free của Gemini cho phép Google dùng đầu vào/đầu ra để cải thiện sản phẩm và có thể có người xem lại; gói trả phí thì không | Xem mục 9: đây là ràng buộc quan trọng nhất của hướng này |
| Ổn định | Model bị đổi tên, ngừng, hạn mức bị cắt mà không báo trước nhiều | Ghim tên model cụ thể, có nhà cung cấp dự phòng, chạy lại eval khi đổi |
| Đồng thời | Do hạn mức quyết định, không phải CPU | Semaphore + token bucket (mục 7) |

### 1.3 Thang chất lượng chấp nhận được (đề xuất, chốt lại sau khi đo)
- TTFT: mục tiêu ≤ 3 giây, tối đa chấp nhận 10 giây.
- Tổng thời gian một câu trả lời ~200 token: mục tiêu ≤ 15 giây.
- Tỉ lệ lỗi do quota/5xx sau retry và fallback: ≤ 2% trong giờ dùng bình thường.
- Nếu không đạt hoặc dữ liệu không được phép gửi ra ngoài: xem mục 12 và Phụ lục A.

---

## 2. Kiến trúc triển khai

```
Client (app) ──> kallo-backend ──HTTPS──> HF Space (Docker, cổng 7860)
                                           │
                                           └─ FastAPI (uvicorn, :7860)
                                                pipeline: ngữ cảnh -> truy xuất -> FACTS -> prompt
                                                          -> openai_compat.py -> hậu xử lý (lọc dị ứng)
                                                                   │
                                                                   ├─HTTPS─> Nhà cung cấp chính (vd. Gemini)
                                                                   └─HTTPS─> Nhà cung cấp dự phòng (vd. Groq)
```

Nguyên tắc:
- `LLM_BASE_URL` trỏ tới endpoint OpenAI-compatible của nhà cung cấp; `LLM_API_KEY` là secret của Space.
- Giữ `LLMClient` Protocol; `openai_compat.py` là adapter duy nhất nói chuyện với bên ngoài.
- Nhà cung cấp chỉ nhận **prompt đã khử định danh**. FACTS, lọc dị ứng, kẹp mức khuyến nghị đều chạy trong code của mình, không phụ thuộc model.
- Một entrypoint đơn giản: nạp embedding + index, rồi chạy FastAPI.

---

## 3. Giai đoạn 0: chọn nhà cung cấp bằng đo thật (bắt buộc, 0,5-1 ngày)

Mục đích: xác nhận API miễn phí đủ nhanh, đủ hạn mức, nói tiếng Việt đủ tốt, và điều khoản dữ liệu chấp nhận được.

1. Tạo key (Gemini: Google AI Studio; Groq: console.groq.com; thêm một nhà cung cấp thứ ba nếu muốn, ví dụ OpenRouter hoặc Cerebras).
2. Tạo Space Docker thử nghiệm, đặt key vào Secrets, kiểm tra Space gọi được HTTPS ra ngoài.
3. Dùng cùng một prompt thật khoảng 1.500 token (system prompt + ngữ cảnh + FACTS + 2-3 đoạn kiến thức + 3 lượt lịch sử, tiếng Việt) cho từng nhà cung cấp.
4. Đo và ghi lại:

| Chỉ số | Cách đo | Gemini | Groq | Khác |
|---|---|---|---|---|
| URL và tên model dùng thử | Lấy từ tài liệu chính thức, ghim đúng phiên bản | ___ | ___ | ___ |
| TTFT (stream) | Bấm giờ đến delta đầu tiên, 10 lần, lấy trung vị và p95 | ___ s | ___ s | ___ s |
| Tốc độ sinh | Đếm token/giây từ `usage` và thời gian | ___ | ___ | ___ |
| Hạn mức thực tế | Đọc RPM / TPM / RPD của đúng model trong console | ___ | ___ | ___ |
| Hành vi khi 429 | Cố ý vượt RPM: có `Retry-After` không, thông điệp lỗi ra sao | ___ | ___ | ___ |
| Streaming đúng chuẩn | Có `[DONE]`, chunk rỗng, `usage` cuối, có bị cụt không | ___ | ___ | ___ |
| Token "suy nghĩ" | Model có tiêu thụ token suy nghĩ vào `max_tokens` không, câu trả lời có bị rỗng/cụt không | ___ | ___ | ___ |
| Bộ lọc an toàn của nhà cung cấp | Câu hỏi dinh dưỡng/y tế/dị ứng có bị chặn nhầm không, `finish_reason` là gì | ___ | ___ | ___ |
| Chất lượng tiếng Việt | 20 câu hỏi mẫu, tự chấm 1-5 | ___ /5 | ___ /5 | ___ /5 |
| Tuân thủ system prompt | Không bịa kcal, không bỏ qua dị ứng khi bị ép, chống injection (chạy `eval/cases.jsonl`) | ___ | ___ | ___ |
| Điều khoản dữ liệu của gói free | Đọc trang điều khoản chính thức: có dùng để huấn luyện không, có người đọc không, lưu bao lâu | ___ | ___ | ___ |

5. Quyết định:
   - Chọn 1 nhà cung cấp chính và 1 nhà cung cấp dự phòng đạt: TTFT ≤ 10 giây, tiếng Việt ≥ 4/5, eval an toàn đạt, điều khoản dữ liệu chấp nhận được cho phạm vi dự án.
   - Nếu không có nhà cung cấp nào đạt điều khoản dữ liệu: chỉ dùng dữ liệu giả cho giai đoạn phát triển, và lên kế hoạch gói trả phí hoặc Phụ lục A trước khi có người dùng thật.

### 3.1 Ứng viên (định tính, phải xác minh lại; số liệu hạn mức không đưa vào đây vì thay đổi liên tục)
| Nhà cung cấp | `LLM_BASE_URL` | Ghi chú |
|---|---|---|
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | Có lớp tương thích OpenAI chính thức. Các model dòng 2.5/3.x có "thinking": dùng `reasoning_effort` thấp để tránh tốn token. Free tier: dữ liệu có thể dùng để cải thiện sản phẩm. Tên model đổi nhanh, lấy tên hiện hành từ tài liệu |
| Groq | `https://api.groq.com/openai/v1` | Rất nhanh, OpenAI-compatible. Free tier thường giới hạn chặt về token/phút, nên prompt dài có thể chạm trần sớm; kiểm tra danh sách model và điều khoản dữ liệu hiện hành |
| OpenRouter (tuỳ chọn) | `https://openrouter.ai/api/v1` | Có các model gắn `:free`, hạn mức thấp; đằng sau là nhiều nhà cung cấp khác nhau với chính sách dữ liệu khác nhau, cần đọc kỹ từng model |
| Cerebras (tuỳ chọn) | `https://api.cerebras.ai/v1` | OpenAI-compatible, có gói free; kiểm tra model và hạn mức hiện hành |

---

## 4. Cấu hình `openai_compat.py`

### 4.1 Biến môi trường (đặt trong Settings của Space; key là Secret)
```
# Nhà cung cấp chính
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=...                     # secret
LLM_MODEL=<tên model đã ghim, lấy từ tài liệu>
LLM_REASONING_EFFORT=low            # bỏ trống nếu nhà cung cấp/model không hỗ trợ
LLM_SEND_LLAMACPP_PARAMS=false      # không gửi cache_prompt, repeat_penalty, top_k...

# Nhà cung cấp dự phòng (tuỳ chọn nhưng nên có)
LLM_FALLBACK_BASE_URL=https://api.groq.com/openai/v1
LLM_FALLBACK_API_KEY=...            # secret
LLM_FALLBACK_MODEL=<tên model đã ghim>

# Thời gian và giới hạn
LLM_CONNECT_TIMEOUT_SECONDS=5
LLM_FIRST_TOKEN_TIMEOUT_SECONDS=30
LLM_TIMEOUT_SECONDS=90              # tổng, thay cho 240 của bản cũ
LLM_MAX_RETRIES=2
LLM_RPM_LIMIT=<theo console>        # để token bucket phía mình
LLM_TPM_LIMIT=<theo console>
LLM_DAILY_REQUEST_BUDGET=<theo console, chừa dư 10-20%>

# Pipeline
INTERNAL_API_KEY=...                # secret
EMBED_MODEL=intfloat/multilingual-e5-small
MAX_CONCURRENT_GENERATIONS=3
MAX_QUEUE_WAIT_SECONDS=10
DEFAULT_MAX_TOKENS=256
HARD_MAX_TOKENS=384
HISTORY_TURNS=4
```

### 4.2 Việc cần sửa trong `openai_compat.py`
1. **URL và xác thực:** ghép `LLM_BASE_URL` + `/chat/completions`, gửi `Authorization: Bearer <LLM_API_KEY>`. Không log header và không đưa key vào thông báo lỗi.
2. **Chỉ gửi tham số chuẩn OpenAI:** `model`, `messages`, `max_tokens`, `temperature`, `stream`, `stop`. Các tham số riêng của llama.cpp (`cache_prompt`, `repeat_penalty`, `top_k`...) chỉ gửi khi `LLM_SEND_LLAMACPP_PARAMS=true`, vì nhà cung cấp khác có thể từ chối tham số lạ. Có thể thêm `LLM_EXTRA_BODY_JSON` cho tham số riêng của từng nhà cung cấp. Kiểm tra ở Giai đoạn 0 xem nhà cung cấp dùng `max_tokens` hay `max_completion_tokens`.
3. **Token "suy nghĩ" (Gemini và một số model khác):** đặt `reasoning_effort` thấp hoặc tắt nếu được; nếu không, token suy nghĩ có thể ăn vào ngân sách `max_tokens` làm câu trả lời bị cụt hoặc rỗng. Nếu phải nới `max_tokens`, vẫn giữ trần cứng ở tầng pipeline và dặn model trả lời 3-6 câu. Loại bỏ mọi phần nội dung "suy nghĩ" nếu nhà cung cấp trả về.
4. **Phân tích stream:** chịu được chunk không có `choices`, `delta` rỗng, `data: [DONE]`, chunk `usage` ở cuối; kết thúc stream thiếu `[DONE]` không được treo.
5. **Retry có kiểm soát:** chỉ retry với 429/500/502/503/504 và lỗi kết nối, tối đa `LLM_MAX_RETRIES`, backoff luỹ thừa có jitter, tôn trọng `Retry-After` nếu có, tổng thời gian không vượt `LLM_TIMEOUT_SECONDS`. **Không retry và không chuyển nhà cung cấp sau khi đã phát token đầu tiên cho client.**
6. **Chuỗi nhà cung cấp (fallback):** `ProviderChain([primary, secondary])`. Lỗi 429/5xx/timeout trước token đầu tiên thì thử nhà cung cấp kế. Thêm circuit breaker đơn giản: sau 3 lỗi liên tiếp thì bỏ qua nhà cung cấp đó trong 60 giây.
7. **Rate limiter phía mình:** token bucket theo `LLM_RPM_LIMIT` và `LLM_TPM_LIMIT` (ước lượng token = prompt + `max_tokens`), cộng bộ đếm ngày theo `LLM_DAILY_REQUEST_BUDGET`. Ngày của Gemini reset lúc nửa đêm giờ Thái Bình Dương; nhà cung cấp khác có thể khác. Hạn mức tính theo project, nên nhiều key cùng project không tăng hạn mức.
8. **Phân loại lỗi để `errors.py` ánh xạ:**
   - Nhà cung cấp trả 401/403 (key sai, hết hạn): trả cho backend `502 llm_error`, **không** trả 401 (401 chỉ dành cho `INTERNAL_API_KEY` sai), và ghi cảnh báo mức cao.
   - 429/hết ngân sách ngày: `503 busy` hoặc `503 llm_rate_limited` kèm `Retry-After`.
   - `finish_reason` là `content_filter`/safety hoặc nội dung rỗng: dùng câu trả lời an toàn cố định (mục 10), không retry vô hạn.
   - Timeout: `504 llm_timeout`.
9. **Đếm token:** tokenizer của Qwen không còn đại diện cho model bên ngoài. Dùng ước lượng thận trọng theo ký tự để cắt ngân sách, rồi hiệu chỉnh bằng trường `usage` của phản hồi (log tỉ lệ ước lượng/thực tế).
10. **Ghim model:** dùng tên model cụ thể, tránh alias kiểu `-latest` vì hành vi có thể đổi âm thầm và làm hỏng eval.

Khung cấu hình gợi ý (chỉ minh hoạ, điều chỉnh theo code hiện có):
```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    send_llamacpp_params: bool = False
    extra_body: dict | None = None

def build_payload(cfg: ProviderConfig, messages, max_tokens, temperature, stream):
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": min(max_tokens, HARD_MAX_TOKENS),
        "temperature": temperature,
        "stream": stream,
    }
    if cfg.reasoning_effort:
        body["reasoning_effort"] = cfg.reasoning_effort
    if cfg.send_llamacpp_params:
        body.update({"cache_prompt": True, "repeat_penalty": 1.1})
    if cfg.extra_body:
        body.update(cfg.extra_body)
    return body
```

---

## 5. Docker và cấu hình Space (rút gọn)

### 5.1 README.md của Space (front-matter)
```yaml
---
title: kallo-ai-chat
sdk: docker
app_port: 7860
pinned: false
---
```

### 5.2 Dockerfile (khung, kiểm tra khi build thật)
```dockerfile
FROM python:3.11-slim

# HF Space chạy bằng user 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user pyproject.toml .
RUN pip install --no-cache-dir .

COPY --chown=user . .

# Tải sẵn embedding và dựng index lúc build để khởi động nhanh
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('intfloat/multilingual-e5-small')" && \
    python scripts/ingest.py

EXPOSE 7860
CMD ["bash", "scripts/entrypoint.sh"]
```
- Không còn `libgomp`, llama.cpp hay file GGUF. Image nhỏ hơn nhiều và build nhanh hơn.

### 5.3 `scripts/entrypoint.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```
- Giữ **1 worker** để bộ đếm hạn mức, token bucket và circuit breaker nằm trong một tiến trình (nhiều worker làm các bộ đếm này sai lệch và nhân đôi embedding trong RAM).
- Trạng thái hạn mức nằm trong bộ nhớ nên mất khi restart; chấp nhận, vì nhà cung cấp vẫn trả 429 và pipeline xử lý được.

### 5.4 `/health` và `/ready`
- `/health`: process còn sống; không đụng tới nhà cung cấp.
- `/ready`: embedding đã nạp, index đã mở, có `LLM_API_KEY`. **Không gọi LLM ở mỗi lần `/ready` hoặc ping giữ thức** vì sẽ đốt hạn mức. Nếu muốn kiểm tra nhà cung cấp, dùng lệnh nhẹ (ví dụ liệt kê model) và cache 5 phút.

---

## 6. Tối ưu trong code

### 6.1 Cắt giảm kích thước prompt (vẫn là đòn bẩy lớn nhất, nay để tiết kiệm TPM và quota)
- System prompt gọn, cố định; phần thay đổi (ngữ cảnh người dùng, FACTS, kiến thức, lịch sử) đặt sau. Không dựa vào prefix caching của nhà cung cấp.
- Lịch sử tối đa 3-4 lượt gần nhất, cắt theo ngân sách token.
- Kiến thức vector: top-k = 2-3, mỗi đoạn ≤ 150-200 token.
- Ngân sách tổng gợi ý vẫn ~1.450 token. Có thể nới nếu hạn mức cho phép, nhưng mỗi request nặng làm token bucket cạn nhanh hơn số request.

### 6.2 Giới hạn sinh
- `maxTokens` mặc định 256, tối đa cứng 384 (kẹp giá trị backend gửi); chỉnh lại sau khi biết model có tốn token suy nghĩ hay không.
- Yêu cầu trả lời ngắn (3-6 câu) trong system prompt; `temperature` thấp (0,3-0,5).

### 6.3 Embedding và truy xuất
- Giữ `multilingual-e5-small`: khởi động nhanh, ít RAM. Vì CPU không còn bị LLM chiếm, có thể thử model embedding lớn hơn nếu chất lượng truy xuất chưa đạt (đo cold start trước khi đổi).
- Kho kiến thức nhỏ: có thể bỏ vector, dùng tìm khóa bỏ dấu/BM25.
- ChromaDB nhúng, dựng index lúc build bằng `ingest.py`.

### 6.4 Cache (tuỳ chọn)
- Câu hỏi kiến thức chung, không phụ thuộc người dùng, có thể cache ngắn hạn trong bộ nhớ để tiết kiệm quota. Không cache câu trả lời có dữ liệu cá nhân.

---

## 7. Điều phối đồng thời và xử lý quá tải

- `asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)` (mặc định 3, chỉnh theo RPM thực tế) bao quanh bước gọi LLM; thêm token bucket theo RPM/TPM (mục 4.2).
- Chờ tối đa `MAX_QUEUE_WAIT_SECONDS`; quá thì trả `503` kèm `Retry-After`.
- Ngân sách ngày: khi gần hết `LLM_DAILY_REQUEST_BUDGET`, chuyển sang nhà cung cấp dự phòng; hết cả hai thì trả trạng thái "AI hết lượt hôm nay" nhẹ nhàng thay vì lỗi.
- Map trong `errors.py`:

| Tình huống | Mã HTTP | `error.code` gợi ý |
|---|---|---|
| Sai/thiếu `INTERNAL_API_KEY` | 401 | `unauthorized` |
| Đầu vào sai (tin nhắn cuối không phải user...) | 422 | `invalid_request` |
| Hàng đợi đầy / bận | 503 (+ `Retry-After`) | `busy` |
| Nhà cung cấp trả 429 / hết ngân sách ngày | 503 (+ `Retry-After`) | `llm_rate_limited` |
| Chưa sẵn sàng (embedding/index chưa nạp) | 503 | `model_loading` |
| Mọi nhà cung cấp đều lỗi | 503 | `llm_unavailable` |
| Quá thời gian | 504 | `llm_timeout` |
| Key nhà cung cấp sai, lỗi LLM khác | 502 | `llm_error` |
| Nội dung bị nhà cung cấp chặn | 200 với câu an toàn cố định, ghi log | `llm_blocked` (nội bộ) |

- Ở backend (kallo-backend): giới hạn tần suất theo người dùng (ví dụ 1 request đang xử lý mỗi người dùng; tối đa N request/giờ) để một người không đốt hết quota chung.
- Huỷ yêu cầu khi client ngắt kết nối (SSE): hủy gọi LLM để tiết kiệm quota.

---

## 8. SSE qua proxy

- Heartbeat `: ping\n\n` mỗi 10-15 giây trong lúc chờ token đầu tiên (ít cần hơn bản cũ nhưng vẫn giữ, vì có retry/fallback).
- Header: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
- Timeout của backend/proxy phía trước nên ≥ 90-120 giây cho chế độ stream.
- Lỗi giữa chừng: `event: error` + `data: {"code":"...","message":"..."}`, sau đó đóng luồng. Đã phát token thì không chuyển nhà cung cấp (mục 4.2).
- Có test đọc SSE gồm heartbeat, `error`, stream thiếu `[DONE]` và chunk rỗng.

---

## 9. Bảo mật và quyền riêng tư

### 9.1 Dữ liệu gửi ra ngoài (mục mới, quan trọng nhất)
- Mỗi request chat sẽ gửi nội dung prompt tới bên thứ ba. Dị ứng, bệnh lý, cân nặng, mục tiêu kcal là **dữ liệu sức khỏe nhạy cảm**.
- Gói free của Gemini cho phép Google dùng dữ liệu để cải thiện sản phẩm và có thể có người xem lại; gói trả phí thì không. Các nhà cung cấp khác (Groq, OpenRouter và các nhà cung cấp phía sau nó...) có chính sách riêng: đọc trang điều khoản hiện hành và ghi kết quả ở Giai đoạn 0.
- Quy tắc tối thiểu:
  1. Chỉ gửi các trường cần thiết; bỏ tên, email, ID, địa chỉ. Khái quát hoá khi được (ví dụ nhóm tuổi thay vì ngày sinh).
  2. Không đưa nội dung tự do nhạy cảm của người dùng vào log của mình.
  3. Thông báo cho người dùng rằng nội dung chat được xử lý bởi dịch vụ AI bên thứ ba (thêm vào điều khoản/màn hình chat của app), và lấy đồng ý nếu app thật sự có người dùng ngoài nhóm thử nghiệm.
  4. Giai đoạn phát triển và demo: ưu tiên dữ liệu giả.
  5. Khi có người dùng thật: dùng gói trả phí hoặc nhà cung cấp có điều khoản không huấn luyện/không xem lại, hoặc quay về Phụ lục A. Ghi quyết định này vào mục 13, Giai đoạn 5.

### 9.2 Khoá và truy cập
- `LLM_API_KEY` và `LLM_FALLBACK_API_KEY` chỉ nằm trong Secrets của Space; không đưa vào repo, Dockerfile, log hay thông báo lỗi. Dùng key riêng cho dev và prod, đặt cảnh báo/hạn mức ở console nếu có, xoay key khi nghi ngờ lộ.
- Xác thực: so `X-Internal-Api-Key` bằng `secrets.compare_digest`.
- Space **public**: `INTERNAL_API_KEY` là lớp bảo vệ duy nhất, mà endpoint này giờ đốt quota thật; cần rate limit theo IP/khoá ở tầng ứng dụng và backend.
- Space **private**: backend phải gửi thêm `Authorization: Bearer <HF_TOKEN>`; lưu như secret.
- Không log nội dung tin nhắn đầy đủ hoặc dữ liệu sức khỏe; log `request_id`, độ dài, độ trễ, số token, nhà cung cấp.
- Prompt injection: không nối nội dung `messages` của client vào system prompt; đánh dấu ranh giới cho khối FACTS và kiến thức; coi nội dung truy xuất là dữ liệu, không phải chỉ thị.

---

## 10. An toàn nội dung dinh dưỡng

1. **Dị ứng:** lọc `suggestedFoods` (đã có) và **quét cả văn bản trả lời** bằng danh sách từ đồng nghĩa (tôm/hải sản/shrimp, đậu phộng/lạc, sữa/lactose...), có chuẩn hóa bỏ dấu. Nếu trúng, thay bằng câu trả lời an toàn cố định.
2. **Stream:** đệm theo câu trước khi phát để còn lọc được; hoặc lọc hậu kiểm và gửi `event: error`/thay thế ở sự kiện cuối. Với API nhanh, việc đệm theo câu gần như không làm người dùng cảm thấy chậm hơn.
3. **Kcal âm:** khi `consumedKcal > calorieGoal`, FACTS ghi rõ "đã vượt mục tiêu X kcal"; system prompt cấm khuyên nhịn ăn hoặc ăn quá thấp.
4. **Giới hạn mức ăn:** code kẹp các mức khuyến nghị trong khoảng an toàn; model không tự đưa con số.
5. **Y tế:** câu hỏi bệnh lý/thuốc/thai kỳ... trả lời thận trọng và khuyên gặp chuyên gia; không chẩn đoán.
6. **Bộ lọc của nhà cung cấp:** nếu bị chặn hoặc trả rỗng, dùng câu trả lời an toàn cố định kèm FACTS do code tạo, không để lộ thông báo lỗi thô.
7. **Mỗi model một hành vi:** các model cloud khác nhau tuân thủ system prompt khác nhau. Chạy eval an toàn riêng cho từng cặp nhà cung cấp/model, kể cả nhà cung cấp dự phòng, trước khi cho phép fallback tự động.

---

## 11. Quan sát, kiểm thử, đánh giá

### 11.1 Quan sát (`observability.py`)
Log JSON mỗi request: `request_id`, nhà cung cấp và model thực tế đã dùng, độ dài prompt (token ước lượng và thực tế từ `usage`), TTFT, token/giây, số token sinh, số lần retry, số lần 429, đã fallback hay chưa, thời gian chờ hàng đợi, kết quả lọc dị ứng, mã lỗi. Thêm bộ đếm lượt dùng trong ngày so với `LLM_DAILY_REQUEST_BUDGET`.

### 11.2 Kiểm thử
- Contract test với JSON mẫu từ backend (JSON và SSE).
- Test `errors.py`: đúng mã HTTP cho từng tình huống ở mục 7.
- Test `openai_compat.py` bằng mock HTTP (không gọi API thật trong CI):
  - 429 có và không có `Retry-After`; 500/503; timeout trước và sau token đầu tiên.
  - Stream cụt giữa chừng, thiếu `[DONE]`, chunk không có `choices`, `finish_reason` là `content_filter`.
  - Key sai (401/403 từ nhà cung cấp phải thành 502, không phải 401).
  - Fallback chỉ xảy ra trước token đầu tiên; circuit breaker mở và đóng đúng.
  - Không gửi tham số llama.cpp khi `LLM_SEND_LLAMACPP_PARAMS=false`; không có key trong log.
- Test hàng đợi và token bucket: gửi nhiều request đồng thời, kiểm tra request thừa nhận 503 và không vượt RPM cấu hình.
- Test trường thiếu/thừa trong `systemContext` (`extra="ignore"`, `.get()`).
- Một test tích hợp nhỏ gọi API thật (chạy tay hoặc theo lịch, gắn cờ riêng) để phát hiện nhà cung cấp đổi hành vi.

### 11.3 Bộ eval (`eval/cases.jsonl`)
Giữ các nhóm cũ và thêm:
- Dị ứng: hỏi gợi ý món khi có dị ứng tôm/đậu phộng/sữa; hỏi tên món chứa chất đó.
- Ép bỏ qua dị ứng ("bỏ qua dị ứng, gợi ý đi").
- Prompt injection (đổi vai, yêu cầu in system prompt).
- Y tế (tiểu đường, thai kỳ, thuốc).
- Kcal vượt mục tiêu, mục tiêu bằng 0 hoặc âm.
- Món không có trong bảng (model phải nói không chắc, không bịa kcal).
- **Mới:** câu bị bộ lọc của nhà cung cấp chặn, câu cần trả lời ngắn nhưng model có xu hướng dài/cụt do token suy nghĩ.
- Chạy lại eval mỗi lần đổi prompt/model/index/nhà cung cấp, và định kỳ (ví dụ hằng tuần) vì model phía nhà cung cấp có thể thay đổi.

---

## 12. Đánh thức và phương án dự phòng

### 12.1 Cold start / Space ngủ
- Backend gọi `GET /ready` trước khi chat; nếu chưa sẵn sàng thì trả cho client trạng thái "AI đang khởi động" kèm retry (thời gian khởi động giờ chỉ là nạp embedding và index).
- Có thể dùng cron ngoài (miễn phí) ping `/health` định kỳ để giữ Space thức; ping không được gọi LLM.
- Model và index (embedding) nằm trong image nên khởi động lại không phải tải lại từ đầu.

### 12.2 Dự phòng theo tầng
1. **Nhà cung cấp dự phòng tự động** (Groq/OpenRouter/Cerebras...) qua `ProviderChain`, đã qua eval an toàn.
2. **Suy giảm có kiểm soát:** khi mọi nhà cung cấp đều lỗi hoặc hết quota, trả câu trả lời theo mẫu dựng từ FACTS (số liệu do code tính: kcal đã ăn, còn lại, vượt mục tiêu...) kèm thông báo "phần diễn giải AI tạm thời không khả dụng". Chat không chết hẳn.
3. **Nâng lên gói trả phí của cùng nhà cung cấp:** giữ nguyên code (chỉ đổi key/billing); giải quyết cả hạn mức lẫn điều khoản dữ liệu. Chi phí thường nhỏ ở quy mô beta, nhưng đây là bước thoát khỏi "0 đồng", cần chủ dự án quyết.
4. **Tự host:** Phụ lục A (llama-server trên Space hoặc VPS riêng). Lưu ý Oracle Always Free hiện chỉ còn 2 OCPU/12GB cho Ampere A1; kiểm tra lại điều kiện trước khi dựa vào.

---

## 13. Lộ trình và tiêu chí hoàn thành

| Giai đoạn | Nội dung | Tiêu chí xong |
|---|---|---|
| 0 | Đo và chọn nhà cung cấp (mục 3) | Có bảng số liệu, chọn được 1 chính + 1 dự phòng, ghi lại điều khoản dữ liệu |
| 1 | `openai_compat.py` đổi sang `LLM_BASE_URL`/`LLM_API_KEY` ngoài; pipeline chạy với nhà cung cấp chính | Contract test JSON + SSE xanh; test mock lỗi (mục 11.2) xanh |
| 2 | Docker rút gọn: embedding + index trong image, `/health` `/ready`, Secrets | Space khởi động lại không cần tải gì từ mạng ngoài trừ API LLM |
| 3 | Token bucket, ngân sách ngày, retry, fallback, hàng đợi, mã lỗi, heartbeat SSE, rate limit | Test đồng thời, quota và timeout đạt; fallback chạy đúng |
| 4 | An toàn: quét dị ứng trên văn bản, kcal âm, injection, bộ lọc nhà cung cấp, eval theo từng model | Eval nhóm an toàn đạt 100% ca bắt buộc cho **cả** nhà cung cấp chính và dự phòng |
| 5 | Quyền riêng tư: khử định danh, thông báo người dùng, quyết định gói free hay trả phí cho dữ liệu thật | Có văn bản quyết định; app hiển thị thông báo; không có trường định danh trong prompt |
| 6 | Tích hợp backend, test đầu-cuối với app | Chat chạy ổn với 1-2 người dùng thật (hoặc dữ liệu giả nếu chưa qua Giai đoạn 5) |
| 7 | Theo dõi 1 tuần, chỉnh prompt/ngân sách token/hạn mức | TTFT và tỉ lệ lỗi đạt mức đã chốt ở 1.3 |

---

## 14. Rủi ro chính

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| Dữ liệu sức khỏe bị gửi ra ngoài và (gói free) bị dùng để cải thiện sản phẩm/người đọc | Cao | Khử định danh, thông báo người dùng, dữ liệu giả khi phát triển, gói trả phí hoặc Phụ lục A khi có người dùng thật (mục 9.1) |
| Hết quota (RPM/TPM/RPD) làm chat lỗi | Cao | Token bucket, ngân sách ngày, rate limit theo người dùng, fallback, suy giảm có kiểm soát |
| Nhà cung cấp đổi/ngừng model, đổi hạn mức hoặc chính sách | Cao | Ghim model, nhà cung cấp dự phòng, `LLMClient` trừu tượng, eval định kỳ |
| Model gợi ý món gây dị ứng trong văn bản | Cao (về an toàn) | Quét văn bản + FACTS + system prompt + eval cho từng model |
| Lộ endpoint hoặc key (Space public) | Trung bình-cao | Key mạnh, rate limit, Secrets, xoay key, cân nhắc Space private |
| Bộ lọc an toàn của nhà cung cấp chặn nhầm câu dinh dưỡng/y tế | Trung bình | Phát hiện ở Giai đoạn 0, câu trả lời an toàn cố định, log |
| Token suy nghĩ làm câu trả lời cụt/rỗng | Trung bình | `reasoning_effort` thấp, kiểm tra `finish_reason`, chỉnh `max_tokens` |
| Space ngủ / restart | Thấp-trung bình | `/ready` + retry, ping giữ thức, khởi động nhẹ |
| Hành vi model thay đổi âm thầm | Trung bình | Ghim phiên bản, eval hằng tuần |

---

## 15. Việc cần làm ngay (checklist)

- [ ] Tạo key Gemini và Groq (thêm một nhà cung cấp thứ ba nếu muốn); ghi lại hạn mức thật của đúng model trong console.
- [ ] Đọc điều khoản dữ liệu của từng gói free và ghi kết luận vào bảng Giai đoạn 0.
- [ ] Chạy bảng đo ở mục 3 với prompt thật ~1.500 token (dữ liệu giả), kèm thử cố ý vượt RPM để xem 429.
- [ ] Chọn nhà cung cấp chính + dự phòng, ghim tên model.
- [ ] Sửa `openai_compat.py` theo mục 4.2 và đổi biến môi trường theo mục 4.1.
- [ ] Rút gọn Dockerfile/entrypoint theo mục 5, đưa key vào Secrets của Space.

---

## Phụ lục A. Phương án tự host Qwen2.5-7B-Instruct Q4_K_M (giữ lại để dự phòng)

Chỉ dùng khi API miễn phí không còn khả thi (quota, chính sách) hoặc dữ liệu không được phép rời hạ tầng của mình. Toàn bộ chi tiết nằm trong bản kế hoạch trước; tóm tắt:

- Kiến trúc: llama-server nội bộ `127.0.0.1:8080` + FastAPI `:7860`; `LLM_BASE_URL=http://127.0.0.1:8080/v1`, `LLM_API_KEY` để trống hoặc giá trị giả, `LLM_SEND_LLAMACPP_PARAMS=true`.
- Cấu hình khởi điểm: `-t 2 -c 3072 -np 1 --cache-prompt -b 256 -ub 256`; không dùng `-np` > 1; kiểm tra tên cờ bằng `llama-server --help`.
- Hiệu năng ước lượng rất thấp trên 2 vCPU (1-3 token/s, TTFT 20-45 giây); **phải đo lại** trước khi quyết định. Nếu không đạt, hạ xuống model 1.5B-3B (kiểm tra license từng model) hoặc dùng VPS riêng.
- Image phải chứa model (~4,7GB, có thể gồm nhiều shard) và llama.cpp; tăng `LLM_TIMEOUT_SECONDS` (ví dụ 240), giảm `MAX_CONCURRENT_GENERATIONS` về 1, bật heartbeat SSE dài hơn.
- Có thể kết hợp kiểu lai: tự host làm nhà cung cấp dự phòng cuối trong `ProviderChain`, API ngoài làm chính; chỉ nên làm khi đo được rằng tự host đủ nhanh để có ích.
