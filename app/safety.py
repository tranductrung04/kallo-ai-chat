import re
import unicodedata


_REDIRECTS = {
    "emergency": ("Nếu bạn đang gặp tình huống nguy cấp hoặc có triệu chứng nặng, hãy gọi dịch vụ cấp cứu địa phương hoặc đến cơ sở y tế gần nhất ngay.", "safety_redirect"),
    "self_harm": ("Mình rất tiếc bạn đang phải chịu điều này. Nếu bạn có nguy cơ làm hại bản thân, hãy gọi dịch vụ khẩn cấp địa phương và liên hệ ngay một người đáng tin cậy; đừng ở một mình.", "safety_redirect"),
    "eating_disorder": ("Mình không thể hỗ trợ nhịn ăn hoặc hành vi ăn uống cực đoan. Hãy trao đổi với bác sĩ hoặc chuyên gia sức khỏe tâm thần/dinh dưỡng để nhận hỗ trợ an toàn.", "safety_redirect"),
}

_PATTERNS = {
    "emergency": (r"cap cuu", r"khong tho", r"dau nguc", r"bat tinh", r"ngat xiu"),
    "self_harm": (r"tu sat", r"tu lam hai", r"lam hai ban than", r"chet di", r"muon chet"),
    "eating_disorder": (r"nhin an", r"an cuc doan", r"an it nhat co the", r"bo bua de giam can", r"non de giam can"),
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def safety_redirect(text: str) -> tuple[str, str] | None:
    folded = _fold(text)
    for category, patterns in _PATTERNS.items():
        if any(re.search(pattern, folded) for pattern in patterns):
            return _REDIRECTS[category]
    return None
