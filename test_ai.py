import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    headers={
        "X-API-Key": "local-dev-7f3a9c2e8b1d"
    },
    json={
        "message": "Привет! Расскажи, что ты умеешь.",
        "system_prompt": "Ты полезный ассистент.",
        "temperature": 0.3
    }
)

response.raise_for_status()
payload = response.json()

print(payload["response"])
print("\nМетрики модели:")
for name, value in payload["metrics"].items():
    print(f"{name}: {value}")
