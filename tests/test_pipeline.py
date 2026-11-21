from app.models import ChatRequest
from app.pipeline import build_prompt, has_allergy_term, sanitize_response


def test_prompt_keeps_facts_and_last_user_message():
    request = ChatRequest.model_validate({
        "messages": [{"role": "user", "content": "Tôi nên ăn gì?"}],
        "systemContext": {"calorieGoal": 1800, "consumedKcal": 1900},
    })
    system, messages = build_prompt(request)
    assert '"calorieGoal": 1800' in system
    assert "Đã vượt mục tiêu 100 kcal" in system
    assert messages[-1]["role"] == "user"


def test_allergy_filter_normalizes_accents_and_synonyms():
    assert has_allergy_term("Món có shrimp", ["tôm"])
    assert "safe" not in sanitize_response("Món có tôm", ["tôm"])
