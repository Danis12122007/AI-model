import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_service.config import Settings
from ai_service.main import create_app


API_KEY = "test-api-key"


def create_client(handler: Callable[[httpx.Request], httpx.Response]) -> TestClient:
    transport = httpx.MockTransport(handler)
    app = create_app(
        Settings(
            api_key=API_KEY,
            ollama_base_url="http://ollama.test",
            model_name="qwen3:4b",
            model_request_timeout=1,
        ),
        http_transport=transport,
    )
    return TestClient(app)


def ollama_chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 25,
            "eval_count": 10,
            "total_duration": 100,
            "load_duration": 20,
            "eval_duration": 80,
        },
    )


def test_chat_requires_api_key() -> None:
    with create_client(lambda _: ollama_chat_response("hello")) as client:
        response = client.post("/api/v1/chat", json={"message": "Привет"})

    assert response.status_code == 401


def test_app_refuses_to_start_without_api_key() -> None:
    app = create_app(
        Settings(api_key=None),
        http_transport=httpx.MockTransport(
            lambda _: ollama_chat_response("unused")
        ),
    )

    with pytest.raises(RuntimeError, match="API_KEY must be configured"):
        with TestClient(app):
            pass


def test_chat_sends_configured_model_options() -> None:
    seen_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(json.loads(request.content))
        return ollama_chat_response("Здравствуйте!")

    with create_client(handler) as client:
        response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": API_KEY},
            json={
                "message": "Привет",
                "system_prompt": "Отвечай кратко",
                "temperature": 0.6,
            },
        )

    assert response.status_code == 200
    assert response.json()["response"] == "Здравствуйте!"
    assert response.json()["metrics"]["done"] is True
    assert response.json()["metrics"]["done_reason"] == "stop"
    assert response.json()["metrics"]["prompt_eval_count"] == 25
    assert response.json()["metrics"]["eval_count"] == 10
    assert seen_requests[0]["model"] == "qwen3:4b"
    assert seen_requests[0]["options"]["num_ctx"] == 4096
    assert seen_requests[0]["options"]["num_thread"] == 8
    assert seen_requests[0]["options"]["temperature"] == 0.6


def test_generate_returns_validated_structured_order_evaluation() -> None:
    result = {
        "category": "веб-разработка",
        "complexity": "средняя",
        "estimated_hours": {"minimum": 12, "maximum": 20},
        "risks": ["Не определён объём тестирования"],
        "missing_requirements": ["Требования к адаптивности"],
        "estimated_cost": {"minimum": 500, "maximum": 900, "currency": "USD"},
        "assumptions": ["Оценка дана для одного разработчика"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["format"]["properties"]["estimated_hours"]["$ref"]
        assert payload["options"]["num_predict"] == 256
        return ollama_chat_response(json.dumps(result, ensure_ascii=False))

    with create_client(handler) as client:
        response = client.post(
            "/api/v1/generate",
            headers={"X-API-Key": API_KEY},
            json={
                "task": "freelance_order_evaluation",
                "data": {"description": "Создать сайт-визитку"},
                "parameters": {"num_predict": 256},
            },
        )

    assert response.status_code == 200
    assert response.json()["result"] == result


def test_generate_rejects_invalid_model_output() -> None:
    invalid_result = {
        "category": "веб-разработка",
        "complexity": "невозможно оценить",
        "estimated_hours": {"minimum": 20, "maximum": 10},
        "risks": [],
        "missing_requirements": [],
        "estimated_cost": None,
        "assumptions": [],
    }

    with create_client(
        lambda _: ollama_chat_response(json.dumps(invalid_result, ensure_ascii=False))
    ) as client:
        response = client.post(
            "/api/v1/generate",
            headers={"X-API-Key": API_KEY},
            json={
                "task": "freelance_order_evaluation",
                "data": {"description": "Оценить заказ"},
            },
        )

    assert response.status_code == 502
    assert "response schema" in response.json()["detail"]


def test_health_reports_model_availability() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:4b", "size": 2_500_000_000}]},
        )

    with create_client(handler) as client:
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": API_KEY},
        )

    assert response.status_code == 200
    assert response.json()["model_available"] is True


def test_generate_supports_greeting_task() -> None:
    greeting = {"text": "С днём рождения, Анна! Пусть этот день будет радостным."}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"].startswith(
            "Ты — автор персональных поздравлений"
        )
        assert json.loads(payload["messages"][1]["content"])["recipient_name"] == "Анна"
        return ollama_chat_response(json.dumps(greeting, ensure_ascii=False))

    with create_client(handler) as client:
        response = client.post(
            "/api/v1/generate",
            headers={"X-API-Key": API_KEY},
            json={
                "task": "greeting",
                "data": {
                    "occasion": "день рождения",
                    "recipient_name": "Анна",
                    "tone": "тёплый",
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["result"] == greeting


def test_health_returns_unavailable_when_model_is_missing() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    with create_client(handler) as client:
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": API_KEY},
        )

    assert response.status_code == 503
    assert response.json()["model_available"] is False


def test_models_lists_ollama_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:4b",
                        "size": 2_500_000_000,
                        "modified_at": "2026-01-01T00:00:00Z",
                    }
                ]
            },
        )

    with create_client(handler) as client:
        response = client.get(
            "/api/v1/models",
            headers={"X-API-Key": API_KEY},
        )

    assert response.status_code == 200
    assert response.json()[0]["name"] == "qwen3:4b"


def test_ollama_timeout_is_reported_as_gateway_timeout() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request timed out")

    with create_client(handler) as client:
        response = client.post(
            "/api/v1/chat",
            headers={"X-API-Key": API_KEY},
            json={"message": "Привет"},
        )

    assert response.status_code == 504


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"description": ""},
    ],
)
def test_order_generation_validates_task_input(data: dict[str, Any]) -> None:
    with create_client(lambda _: ollama_chat_response("{}")) as client:
        response = client.post(
            "/api/v1/generate",
            headers={"X-API-Key": API_KEY},
            json={"task": "freelance_order_evaluation", "data": data},
        )

    assert response.status_code == 422
