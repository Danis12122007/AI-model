from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen3:4b"
    model_temperature: float = 0.3
    model_context_length: int = 4096
    model_max_tokens: int = 1024
    model_num_threads: int = 8
    model_keep_alive: str = "10m"
    model_request_timeout: float = 180.0
    max_concurrent_requests: int = 1

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("model_temperature")
    @classmethod
    def validate_model_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("model_temperature must be between 0 and 2")
        return value

    @field_validator(
        "model_context_length",
        "model_max_tokens",
        "model_num_threads",
        "max_concurrent_requests",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be greater than zero")
        return value

    @field_validator("model_request_timeout")
    @classmethod
    def validate_request_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("model_request_timeout must be greater than zero")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
