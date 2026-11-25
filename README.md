# Local AI Service

Локальный HTTP-сервис для нескольких приложений. FastAPI обращается к Ollama через HTTP; веса модели хранятся отдельно от API и Docker-образа. В первой версии нет базы данных, истории запросов и облачных AI API.

## Реализовано

- **Этап 1 — основа:** конфигурация через `.env`, ключ API, модели запросов/ответов и клиент Ollama.
- **Этап 2 — сценарии:** `/chat`, реестр задач для `/generate`, отдельные промпты, структурированный и проверяемый Pydantic-ответ для оценки заказов.
- **Этап 3 — запуск и перенос:** Dockerfile API, Compose с отдельным контейнером и volume Ollama, инструкции для виртуального окружения и запусков.

## Требования

- Python 3.12 или новее для локального запуска.
- Ollama с установленной моделью Qwen3 4B. API и Ollama запускаются независимо.
- Для Compose нужен Docker с поддержкой `docker compose`.

Сервис передаёт модель Ollama; он не загружает веса в API-контейнер и не вызывает облачные модели. `qwen3:4b` — настраиваемый тег модели. Проверьте фактический размер и квантование локального артефакта командой `ollama show qwen3:4b`; тег и доступная квантовка могут зависеть от сборки Ollama.

## Локальный запуск

Создайте и активируйте `.venv` из Python 3.12+.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Задайте уникальный секрет `API_KEY` в `.env`. Параметры модели, адрес Ollama и таймауты также настраиваются там. Запустите Ollama отдельно, затем загрузите модель:

```bash
ollama pull qwen3:4b
ollama show qwen3:4b
```

Запускайте приложение из активированного виртуального окружения и из корня проекта:

```bash
uvicorn ai_service.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Для разработки API по умолчанию обращается к `http://localhost:11434`. Не выставляйте API наружу без TLS и надлежащей защиты секрета.

## Docker Compose

```bash
Copy-Item .env.example .env
```

Отредактируйте `.env` и задайте `API_KEY`, затем запустите оба независимых сервиса:

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama show qwen3:4b
```

API доступен на `http://localhost:8000`. Ollama и веса находятся в отдельном контейнере и volume `ollama_data`; для обновления образов API и Ollama используйте соответствующий сервис Compose отдельно. CPU-режим является настройкой по умолчанию; ускорение GPU не требуется. Не публикуйте порт Ollama в сеть без отдельной необходимости и защиты.

## Конфигурация

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_KEY` | обязательна при запуске | Секрет для заголовка `X-API-Key` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Адрес Ollama (в Compose задаётся как `http://ollama:11434`) |
| `MODEL_NAME` | `qwen3:4b` | Имя/тег модели в Ollama |
| `MODEL_TEMPERATURE` | `0.3` | Температура генерации, от 0 до 2 |
| `MODEL_CONTEXT_LENGTH` | `4096` | Контекст модели в токенах |
| `MODEL_MAX_TOKENS` | `1024` | Максимум генерируемых токенов по умолчанию |
| `MODEL_NUM_THREADS` | `8` | Потоки CPU, передаваемые Ollama |
| `MODEL_KEEP_ALIVE` | `10m` | Время удержания модели в памяти Ollama |
| `MODEL_REQUEST_TIMEOUT` | `180` | Таймаут запроса в секундах |
| `MAX_CONCURRENT_REQUESTS` | `1` | Максимум одновременных запросов к модели в одном процессе API |

Установите `OLLAMA_NUM_PARALLEL=1` и `OLLAMA_MAX_LOADED_MODELS=1` для контейнера Ollama (Compose уже задаёт их), чтобы ограничить параллелизм и число загруженных моделей. Для API используйте один Uvicorn worker, иначе ограничение одновременных запросов будет действовать отдельно в каждом процессе.

## HTTP API

Все эндпоинты требуют заголовок `X-API-Key: <API_KEY>`.

- `POST /api/v1/chat` — `{ "message": "...", "system_prompt": "...", "temperature": 0.3 }`
- `POST /api/v1/generate` — `{ "task": "...", "data": {}, "parameters": { "temperature": 0.3, "num_predict": 500 } }`
- `GET /api/v1/health` — доступность Ollama и настроенной модели; возвращает `503`, если Ollama или модель недоступны.
- `GET /api/v1/models` — список установленных в Ollama моделей.

Доступные идентификаторы задач:

- `freelance_order_evaluation` — структурированный JSON с категорией, сложностью, оценкой часов и стоимости, рисками, недостающими требованиями и предположениями.
- `greeting` — генерация поздравления с учётом повода, имени, стиля, длины и тональности.

Пример:

```bash
curl http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -d '{"task":"freelance_order_evaluation","data":{"description":"Разработать адаптивный сайт-каталог"}}'
```

Неизвестный сценарий возвращает `404`, некорректные поля запроса — `422`, таймаут Ollama — `504`, а недоступность Ollama или невалидный ответ модели — `503`/`502`.

## Добавление сценария

Создайте входную и выходную Pydantic-модели, промпт в `src/ai_service/prompts/` и регистрацию `TaskDefinition` в отдельном модуле под `src/ai_service/tasks/`. Подключите модуль в `src/ai_service/tasks/__init__.py`. Общий обработчик `/generate` останется без изменений; выход модели автоматически проверяется заданной Pydantic-схемой.

## Проверки

Тесты запускаются из активированного окружения:

```bash
pytest
```

Тесты используют mock HTTP-транспорт и не требуют запущенного Ollama или скачивания весов. В этой среде тесты выполнены на Python 3.11.5: `12 passed`. Python 3.12+ требуется для запуска приложения; проверить тесты именно на Python 3.12 в этой среде не удалось — доступный каталог Python 3.12 не содержит исполняемого файла. Проверка конфигурации Compose прошла; собрать Docker-образ не удалось, потому что Docker daemon недоступен. Docker-образ и локальные инструкции настроены на Python 3.12+.
