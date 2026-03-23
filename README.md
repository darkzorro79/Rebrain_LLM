# Rebrain LLM — курс по Large Language Models

Репозиторий содержит практические задания курса REBRAIN по работе с LLM.  
Ниже — краткое описание каждого блока и решённых задач.

---

## Блок 03 — Сервер инференса vLLM

**Задача:** развернуть OpenAI-совместимый HTTP-сервер на базе **vLLM** с моделью `Qwen/Qwen2.5-0.5B-Instruct` на GPU-машине (NVIDIA H200).

**Решение:** скрипт `bash03.sh` устанавливает vLLM, запускает сервер в фоне через `nohup` на порту `8000`, настраивает GPU-память, prefix caching и API-ключ. Скрипты `check_models.sh` и `check_chat_completions.sh` проверяют критерии.

---

## Блок 04 — Локальный сервер llama.cpp (CPU-only)

**Задача:** развернуть локальный LLM-сервер с помощью **llama.cpp** (`llama-server`) на CPU-машине без GPU, модель `Qwen2.5-7B-Instruct-Q4_K_M.gguf` на порту `8080`.

**Решение:** скрипт `llama-server-Qwen2.5-7B-Instruct-Q4_K_M.sh` запускает `llama-server` с загрузкой модели напрямую с Hugging Face Hub. Скрипт `check_models.sh` проверяет наличие модели через `/v1/models`.

---

## Блок 05 — Клиент к облачному API (Gemini)

**Задача:** написать Python-скрипт `client.py`, который принимает промпт из аргументов командной строки, отправляет запрос к `gemini-2.5-flash` через `https://llm.rebrainme.study` и выводит только текст ответа в stdout. API-ключ читается из переменной окружения `OPENAI_API_KEY`.

**Решение:** `user-repo-vaemy/client.py` использует `argparse`, `openai` SDK, `os.getenv` для ключа и `python-dotenv` для локального `.env`.

---

## Блок 06 — Fine-Tuning модели с LoRA

**Задача:** дообучить **Qwen2.5-7B-Instruct** в стиле пирата методом **LoRA** на GPU (NVIDIA H100), сохранить адаптер в `/opt/train/qwen_final`, запустить vLLM с поддержкой LoRA.

**Решение:** `train/train.py` — SFT через `TRL SFTTrainer` + `PEFT LoraConfig` (rank 64, alpha 128, BF16). Датасет пиратских диалогов генерируется `dataset.py`. Сервер поднимается через `run_server.sh` с флагами `--enable-lora --lora-modules`.

---

## Блок 07 — Prompt Engineering (парсинг логов)

**Задача:** написать скрипт `client.py`, который считывает неструктурированные текстовые логи и с помощью LLM (`gemini-2.5-flash`) преобразует их в JSON с полями `service_name`, `error_code`, `severity` (CRITICAL/WARNING/INFO), `message`. Обязательно: system prompt с ролью Log Parser, few-shot пример, `response_format={"type": "json_object"}`.

**Решение:** `client.py` скачивает логи по URL, отправляет в `gemini-2.5-flash` через `openai` SDK с few-shot и structured output, выводит JSON в stdout.

---

## Блок 08 — Безопасность LLM (Guardrails)

**Задача:** реализовать двухэтапный пайплайн в скрипте `guardrail.py`: модель-инспектор (`gemini-2.5-flash`) проверяет запрос на Prompt Injection / Jailbreak и возвращает JSON с полями `score`, `verdict`, `comment`. При `score >= 8.0` — выводится сообщение **ATTACK** и выход. Иначе — запрос передаётся основной модели `gemini-3-flash-preview`.

**Решение:** `guardrail.py` использует `response_format` с `json_schema`, `pydantic`-модель `SecurityResult`, две разные модели и порог `8.0`.

---

## Блок 09 — RAG с PostgreSQL и pgvector

**Задача:** реализовать полный RAG-цикл: PostgreSQL (`support_db`) с расширением pgvector, таблица `knowledge_base` с векторами `vector(3072)`. Скрипт `indexer.py` индексирует 5 документов базы знаний техподдержки через `gemini-embedding-001`. Скрипт `rag_chat.py` принимает вопрос, ищет ближайший документ и отвечает через LLM.

**Решение:** `practic/indexer.py` и `practic/rag_chat.py` с поиском `ORDER BY embedding <-> %s::vector`. Docker Compose поднимает PostgreSQL с pgvector.

---

## Блок 10 — RAG по Kubernetes CHANGELOG

**Задача:** реализовать семантический поиск по `CHANGELOG-1.35.md` Kubernetes. Скрипт `indexer.py` скачивает файл, нарезает через `MarkdownTextSplitter`, векторизует через `gemini-embedding-001` и сохраняет в PostgreSQL (`rag_db`). Скрипт `rag.py` ищет 3 ближайших чанка и отвечает через `gemini-2.5-flash`.

**Решение:** `practic/indexer.py` и `practic/rag.py` с поиском `ORDER BY embedding <=>`, минимум 20 чанков в БД.

---

## Блок 11 — Продвинутые RAG-пайплайны

**Задача:** реализовать переранжирование (Re-ranking) через **Cross-Encoder**: скрипт `reranker.py` вычисляет оценки релевантности для трёх документов по запросу «Почему не удаётся подключиться к базе данных?» с помощью модели `cross-encoder/ms-marco-MiniLM-L-6-v2` и выводит документы в порядке убывания score.

**Решение:** `practic/reranker.py` использует `CrossEncoder` из `sentence_transformers`. Дополнительно реализованы **RAG-Fusion** (RRF) и **HyDE** (гипотетический документ).

---

## Блок 12 — LlamaIndex и RAG на PostgreSQL

**Задача:** реализовать RAG-цикл через фреймворк **LlamaIndex**: скрипт `ingest.py` рекурсивно загружает документацию nginx из GitHub и индексирует в PostgreSQL (`support` / pgvector), скрипт `query.py` принимает вопрос и отвечает через LlamaIndex query engine.

**Решение:** `practic/user-repo-bgrht/ingest.py` и `query.py` используют `PGVectorStore`, `OllamaEmbedding`, `SimpleDirectoryReader(..., recursive=True)` и `VectorStoreIndex`.

---

## Блок 13 — Агенты и Function Calling

**Задача:** реализовать помощника системного администратора `helper.py` — бесконечный цикл ввода, модель `gemini-2.5-flash` с tool `run_shell_command`, обязательное подтверждение команды у пользователя (`y`/`n`) перед запуском, передача stdout обратно в модель.

**Решение:** `practic/helper.py` — полный агентный цикл с `subprocess.run`, интерактивным подтверждением и передачей результата в историю с ролью `tool`.

---

## Блок 14 — MCP (Model Context Protocol)

**Задача:** реализовать MCP-сервер `server.py` с инструментом `check_service_status` (запуск `systemctl status <service_name>`) через **FastMCP** и интерактивный клиент `client.py` — цикл ввода, LLM (`gemini-2.5-flash`) с tools из MCP, вызов инструмента при tool call и передача результата обратно в модель.

**Решение:** `practic/server.py` (FastMCP + stdio-транспорт) и `practic/client.py` (OpenAI-совместимый клиент + `session.call_tool`). Есть также HTTP-вариант в корне блока.

---

## Блок 15 — LangChain: Guard + Worker

**Задача:** написать скрипт `guardian.py` на **LangChain**: Guard (`gemini-2.5-flash`) проверяет запрос на инъекции, возвращает JSON `{is_safe, reason}`; если `is_safe=true` — запрос передаётся Worker (`gemini-2.5-pro`); иначе выводится «Запрос отклонен системой безопасности». Бесконечный цикл ввода, этапы обработки наглядно подписаны.

**Решение:** `practic/guardian.py` — две LCEL-цепочки (`ChatPromptTemplate | ChatOpenAI | парсер`) с `JsonOutputParser` для Guard и `StrOutputParser` для Worker.

---

## Блок 16 — n8n: Интеллектуальный дежурный

**Задача:** создать воркфлоу в **n8n**, который получает технические алерты через Telegram, классифицирует их через AI Agent (Guard, `gemini-2.5-flash`), анализирует и формирует структурированный Markdown-отчёт в Telegram с блоками: Тип проблемы, Уровень критичности, Рекомендация. При недоступности AI — отправляет сырой алерт с пометкой «Ошибка анализа: AI недоступен».

**Решение:** `Intelligent_OnCall_Alert_Analyzer.json` — JSON-экспорт воркфлоу с Telegram Trigger, Guard Agent, Worker Agent, ветвлением If, Error Trigger и двумя Telegram-нодами для успешного и ошибочного ответа.

---

*Курс на платформе REBRAIN: [LLM](https://my.rebrainme.com/course/llm)*
