# Блок 10: RAG по Kubernetes CHANGELOG

## Практическое задание

### Ресурсы

Вам предоставлена виртуальная машина с Ubuntu 24.04 с правами root и установленным Python.

### Задача

Реализовать систему семантического поиска по списку изменений (Changelog) Kubernetes v1.35.

### Подготовка инфраструктуры (PostgreSQL)

1. Установите в систему postgresql и расширение pgvector.
2. Создайте базу данных `rag_db`.
3. В базе `rag_db` активируйте расширение `vector`.
4. Создайте таблицу `knowledge_base` со следующими полями:
   - `id`: serial primary key
   - `content`: text (текст чанка)
   - `metadata`: jsonb (для хранения метаданных, если необходимо)
   - `embedding`: vector(3072) (размерность для модели `gemini-embedding-001`).

### Создайте скрипт индексации (`indexer.py`), который

- Загружает файл `CHANGELOG-1.35.md` по ссылке: [https://raw.githubusercontent.com/kubernetes/kubernetes/master/CHANGELOG/CHANGELOG-1.35.md](https://raw.githubusercontent.com/kubernetes/kubernetes/master/CHANGELOG/CHANGELOG-1.35.md).
- Использует `MarkdownTextSplitter` из библиотеки `langchain-text-splitters` для разбиения текста на логические части.
- Генерирует эмбеддинги для каждого чанка через модель `gemini-embedding-001`.
- Сохраняет чанки и их векторы в таблицу `knowledge_base`.

### Скрипт поиска и ответа (`rag.py`)

- Принимает один аргумент командной строки — вопрос пользователя (например: «Что нового в работе с API?»).
- Генерирует эмбеддинг вопроса.
- Выполняет поиск в базе данных (используйте оператор `<->` или `<=>`) для нахождения 3-х наиболее релевантных чанков.
- Отправляет найденный контекст и вопрос в LLM (модель `gemini-2.5-flash`) с промптом: *Используя предоставленные данные из CHANGELOG: {context}, ответь на вопрос: {question}*.
- Выводит ответ в stdout.

### Критерии успешного выполнения

- В Git-репозитории присутствуют файлы `indexer.py` и `rag.py`.
- В таблице `knowledge_base` успешно созданы записи с векторами (не менее 20 чанков).
- Скрипт `rag.py` возвращает корректные ответы на основе содержимого файла (например, находит информацию о «SIG Instrumentation» или «API Changes»).

---

## Описание решения в этой папке

Эталонная реализация задания лежит в каталоге **`practic/`**.

### Основные файлы

| Файл | Назначение |
|------|------------|
| [`practic/indexer.py`](practic/indexer.py) | Скачивает CHANGELOG по URL из задания, режет текст через `MarkdownTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`), для каждого чанка вызывает `gemini-embedding-001` (вектор 3072) через OpenAI-совместимый клиент и пишет строки в `knowledge_base` (`content`, `metadata` как JSONB с индексом чанка и источником). |
| [`practic/rag.py`](practic/rag.py) | Принимает вопрос из `sys.argv`, строит тот же эмбеддинг, выбирает **3** ближайших чанка запросом `ORDER BY embedding <=> %s`, склеивает контекст и передаёт в `gemini-2.5-flash` с промптом в формулировке из задания; ответ печатается в stdout. |
| [`practic/requirements.txt`](practic/requirements.txt) | Зависимости: `psycopg2-binary`, `langchain-text-splitters`, `requests`, `openai`, опционально `python-dotenv` и `google-generativeai` для совместимости окружения. |
| [`practic/.env.example`](practic/.env.example) | Пример переменных: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (по умолчанию учебный endpoint), `DB_DSN` или отдельные `POSTGRES_*`. |
| [`practic/run.sh`](practic/run.sh) | Устанавливает зависимости, проверяет ключи и `DB_DSN`, запускает индексацию. |
| [`practic/base/db/init-pgvector.sql`](practic/base/db/init-pgvector.sql) | Включает `vector` и создаёт таблицу `knowledge_base` с полями из задания. |
| [`practic/base/docker-compose.yml`](practic/base/docker-compose.yml) | Поднимает PostgreSQL с образом, где доступен pgvector; БД `rag_db`, пользователь/пароль согласуются с примером DSN. |

Подробные шаги установки PostgreSQL, настройки `.env` и примеры вызовов `python3 indexer.py` / `python3 rag.py "…"` описаны в [`practic/README.md`](practic/README.md).

### Прочие материалы в `block10/`

В **корне** блока лежат вспомогательные примеры курса (индексация PDF, чанкинг `policy.md` / `.txt`, отдельные `rag_app*.py`, `retriever*.py`, `indexer_ollama.py` и т.д.). Они **не** являются обязательной частью проверки задания по Kubernetes CHANGELOG; для сдачи ориентируйтесь на пару `practic/indexer.py` и `practic/rag.py` плюс развёрнутая БД с pgvector.
