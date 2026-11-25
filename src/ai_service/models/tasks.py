from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str = Field(min_length=1, max_length=20_000)


class EstimateRange(BaseModel):
    minimum: float = Field(ge=0)
    maximum: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "EstimateRange":
        if self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        return self


class CostEstimate(EstimateRange):
    currency: str = Field(min_length=1, max_length=12)


class OrderEvaluation(BaseModel):
    category: str = Field(min_length=1)
    complexity: Literal["низкая", "средняя", "высокая"]
    estimated_hours: EstimateRange
    risks: list[str]
    missing_requirements: list[str]
    estimated_cost: CostEstimate | None
    assumptions: list[str]


class GreetingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occasion: str = Field(min_length=1, max_length=200)
    recipient_name: str | None = Field(default=None, max_length=100)
    style: str = Field(default="дружелюбный", max_length=100)
    length: Literal["short", "medium", "long"] = "medium"
    tone: str = Field(default="тёплый", max_length=100)


class GreetingOutput(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
