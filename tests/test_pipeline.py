from app.models import ChatRequest
from app.pipeline import build_prompt, filter_suggested_foods, has_allergy_term, safe_response, sanitize_response


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
    assert not has_allergy_term("Tóm lại, nên ăn nhẹ", ["tôm"])
    assert has_allergy_term("Tôm hùm và eggs", ["hải sản", "trứng"])
    assert has_allergy_term("Peanuts", ["lạc"])
    assert not has_allergy_term("Lạc quan là tốt", ["lạc"])
    assert sanitize_response("Món có tôm", ["tôm"]) == safe_response(["tôm"])
    assert filter_suggested_foods(["Bún riêu tôm", "Cơm rau"], ["tôm"]) == ["Cơm rau"]


def test_backend_contract_options_and_nested_context():
    request = ChatRequest.model_validate({
        "messages": [{"role": "user", "content": "Còn bao nhiêu calo?"}],
        "systemContext": {"targets": {"dailyKcal": 1800}, "today": {"consumedKcal": 1200}},
        "options": {"stream": True, "maxTokens": 128},
    })
    assert request.effective_stream is True
    assert request.effective_max_tokens == 128
    system, _ = build_prompt(request)
    assert '"remainingKcal": 600' in system
