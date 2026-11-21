from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def alias(name: str) -> str:
    return {
        "system_context": "systemContext",
        "max_tokens": "maxTokens",
        "calorie_goal": "calorieGoal",
        "consumed_kcal": "consumedKcal",
        "suggested_foods": "suggestedFoods",
    }.get(name, name)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class SystemContext(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    calorie_goal: float | None = None
    consumed_kcal: float | None = None
    allergies: list[str] = Field(default_factory=list)
    suggested_foods: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    system_context: SystemContext = Field(default_factory=SystemContext)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("last message must have role user")
        return value


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[dict[str, Any]]


class ErrorBody(BaseModel):
    code: str
    message: str
