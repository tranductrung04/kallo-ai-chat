# Kallo AI Chat

FastAPI service for Kallo nutrition chat. LLM generation uses external OpenAI-compatible providers; Space stores keys only in Secrets.

## Local run

```bash
copy .env.example .env
pip install -e ".[test]"
uvicorn app.main:app --reload --port 7860
```

Required settings: `INTERNAL_API_KEY`, `LLM_API_KEY`, `LLM_MODEL`. Set `LLM_BASE_URL` to provider endpoint without `/chat/completions`.

Optional fallback: `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_API_KEY`, `LLM_FALLBACK_MODEL`.

## API

- `GET /health`: process health, no provider call.
- `GET /ready`: configuration readiness, no provider call.
- `POST /v1/chat`: JSON or SSE; send `X-Internal-Api-Key`.

```json
{
  "messages": [{"role": "user", "content": "Tôi nên ăn gì?"}],
  "systemContext": {
    "calorieGoal": 1800,
    "consumedKcal": 1200,
    "allergies": ["tôm"],
    "suggestedFoods": [{"name": "cơm gạo lứt", "kcal": 320}, {"name": "ức gà", "kcal": 250}]
  },
  "stream": false
}
```

Do not put real health data in free-tier provider requests until data terms are approved. Use synthetic data during development.
