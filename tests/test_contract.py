import json

from app.models import ChatRequest
from app.pipeline import build_prompt, facts_text


def test_backend_context_preserves_today_and_redacts_profile():
    request = ChatRequest.model_validate({
        "messages": [{"role": "user", "content": "Còn bao nhiêu calo?"}],
        "systemContext": {
            "profile": {"displayName": "Người thật", "timezone": "Asia/Ho_Chi_Minh", "heightCm": 170, "goal": "giảm cân"},
            "allergies": ["tôm"],
            "today": {"calorieGoal": 1800, "consumedKcal": 1200, "mealsLogged": 2, "waterMl": 1000, "waterGoalMl": 2000, "steps": 5000, "caloriesBurned": 220},
            "suggestedFoods": [{"name": "Cơm rau", "kcal": 300}, "Ức gà"],
        },
    })
    facts = facts_text(request)
    facts_json = json.loads(facts)
    assert facts_json["remainingKcal"] == 600
    assert facts_json["today"]["mealsLogged"] == 2
    assert facts_json["today"]["waterMl"] == 1000
    assert facts_json["today"]["steps"] == 5000
    assert "Người thật" not in facts
    assert "Asia/Ho_Chi_Minh" not in facts
    system, _ = build_prompt(request)
    assert '"name":"Cơm rau","kcal":300' in system
    assert '"Ức gà"' in system
