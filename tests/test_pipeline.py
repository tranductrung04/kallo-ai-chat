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
    assert not has_allergy_term("Tóm tắt thực đơn và trung bình", ["tôm", "trứng"])
    assert has_allergy_term("Lạc đường", ["lạc"])
    assert has_allergy_term("Tóm lại, hãy ăn tôm rang muối", ["tôm"])
    assert has_allergy_term("Bánh mì và phô mai", ["gluten", "sữa"])
    assert has_allergy_term("Cua, sò, nghêu và hàu", ["hải sản"])
    assert not has_allergy_term("Đây là lời khuyên cá nhân", ["hải sản"])
    assert not has_allergy_term("Nêm chút mì chính", ["gluten"])
    assert has_allergy_term("Đậu phộng rang và bánh mì", ["dau phung", "gluten"])
    assert has_allergy_term("Tom hum xao", ["tom"])
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


def test_seed_group_and_code_allergies_block_dishes_without_false_positives():
    seed_allergies = ["Giáp xác (Tôm, Cua, Ghẹ)", "Đậu phụng (Lạc)", "Sữa và Sản phẩm từ sữa"]
    assert has_allergy_term("bún riêu tôm", seed_allergies)
    assert has_allergy_term("đậu phộng rang", seed_allergies)
    assert has_allergy_term("cua rang me", seed_allergies)
    assert has_allergy_term("Bánh phô mai nướng", seed_allergies)
    assert not has_allergy_term("Mục tiêu của bạn còn 500 kcal, tất cả các món gà đều ổn.", seed_allergies)
    assert not has_allergy_term("Bạn có thể chỉnh sửa một số bữa ăn.", seed_allergies)

    code_allergies = ["CRUSTACEANS", "PEANUT", "DAIRY", "SESAME"]
    assert has_allergy_term("bún riêu tôm", code_allergies)
    assert has_allergy_term("cua rang me", code_allergies)
    assert has_allergy_term("đậu phộng rang", code_allergies)
    assert has_allergy_term("sữa tươi không đường", code_allergies)
    assert not has_allergy_term("cá hồi áp chảo", ["CRUSTACEANS"])
    assert not has_allergy_term("vịt rang me", ["SESAME"])
    assert has_allergy_term("gà rắc mè đen", ["SESAME"])

    object_allergies = [{"code": "CRUSTACEANS", "name": "Giáp xác (Tôm, Cua, Ghẹ)"}]
    assert has_allergy_term("cua rang me", object_allergies)
