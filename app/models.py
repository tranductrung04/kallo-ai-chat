from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def alias(name: str) -> str:
    return {
        "system_context": "systemContext",
        "max_tokens": "maxTokens",
        "calorie_goal": "calorieGoal",
        "consumed_kcal": "consumedKcal",
        "suggested_foods": "suggestedFoods",
        "daily_kcal": "dailyKcal",
        "calories_in": "caloriesIn",
    }.get(name, name)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=12000)


class ChatOptions(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False


class ContextTargets(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    daily_kcal: float | None = None
    calorie_goal: float | None = None
    tdee: float | None = None


class ContextToday(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    consumed_kcal: float | None = None
    calories_in: float | None = None
    protein: float | None = None
    carbs: float | None = None
    fat: float | None = None


class SystemContext(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    calorie_goal: float | None = None
    consumed_kcal: float | None = None
    allergies: list[str] = Field(default_factory=list, max_length=32)
    suggested_foods: list[str] = Field(default_factory=list, max_length=32)
    facts: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    targets: ContextTargets = Field(default_factory=ContextTargets)
    today: ContextToday = Field(default_factory=ContextToday)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=alias)

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    system_context: SystemContext = Field(default_factory=SystemContext)
    options: ChatOptions = Field(default_factory=ChatOptions)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("last message must have role user")
        for message in value:
            if message.role == "user" and len(message.content) > 2000:
                raise ValueError("user message exceeds 2000 characters")
        return value

    @property
    def effective_max_tokens(self) -> int | None:
        return self.options.max_tokens or self.max_tokens

    @property
    def effective_stream(self) -> bool:
        return self.options.stream or self.stream

    @property
    def effective_system_context(self) -> SystemContext:
        context = self.system_context.model_copy(deep=True)
        if context.calorie_goal is None:
            context.calorie_goal = context.targets.daily_kcal
            if context.calorie_goal is None:
                context.calorie_goal = context.targets.calorie_goal
        if context.consumed_kcal is None:
            context.consumed_kcal = context.today.consumed_kcal
            if context.consumed_kcal is None:
                context.consumed_kcal = context.today.calories_in
        return context


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[dict[str, Any]]


class ErrorBody(BaseModel):
    code: str
    message: str
