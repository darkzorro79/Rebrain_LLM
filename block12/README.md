# Блок 12: LlamaIndex и RAG на PostgreSQL (pgvector)

Материалы курса [«LlamaIndex: фреймворк для RAG»](https://my.rebrainme.com/course/llm/task/3153): индексация документов, векторное хранилище в PostgreSQL, запрос с извлечением контекста и генерацией ответа.

## Практическое задание

### Ресурсы

Вам предоставлена виртуальная машина с Ubuntu 24.04 с правами root и установленным Python. А также — выделенный Git-репозиторий, куда будет необходимо загрузить ответ на задание.

### Задача

Вам необходимо реализовать полноценный RAG-цикл (Retrieval Augmented Generation) на базе PostgreSQL с использованием LlamaIndex.

#### PostgreSQL

- Установите в систему postgresql и расширение pgvector.
- Создайте базу данных `support`, пользователя `support` с паролем `password`.
- В базе `support` активируйте расширение: `CREATE EXTENSION IF NOT EXISTS vector;`.

#### Скрипт индексации (`ingest.py`)

Создайте скрипт, который подключается к БД и индексирует документацию nginx (директория `content` в репозитории [https://github.com/nginx/documentation](https://github.com/nginx/documentation)).

Обратите внимание, что индексировать директорию нужно **рекурсивно**. Добавьте нужный параметр для LlamaIndex.

#### Скрипт поиска и ответа (`query.py`)

Создайте скрипт, который:

- Принимает один аргумент командной строки — вопрос пользователя.
- С помощью LlamaIndex ищет подходящие документы и отвечает на вопрос пользователя.

### Критерии успешного выполнения

- В Git-репозитории присутствуют файлы `ingest.py` и `query.py`.
- В базе `support` находится проиндексированная документация nginx.
- Скрипты `ingest.py` и `query.py` используют LlamaIndex для индексации документов и обработки вопросов пользователя.

---

## Описание решения в этой папке

### RAG с PostgreSQL и LlamaIndex (задание курса)

| Файл | Назначение |
|------|------------|
| [`practic/user-repo-bgrht/ingest.py`](practic/user-repo-bgrht/ingest.py) | Подключение к PostgreSQL через `PGVectorStore` (таблица `rag_nginx_knowledge_base`, размерность эмбеддингов 4096 под модель из Ollama). Загрузка документов: `SimpleDirectoryReader("./content", recursive=True).load_data()` — рекурсивный обход каталога. Очистка текста от символов NUL (`\x00`) перед записью в Postgres. Глобальная модель эмбеддингов: `OllamaEmbedding` (`qwen3-embedding:8b-q4_K_M`, `http://localhost:11434`). Сборка индекса: `VectorStoreIndex.from_documents(..., storage_context=...)`. |
| [`practic/user-repo-bgrht/query.py`](practic/user-repo-bgrht/query.py) | Один аргумент CLI — вопрос. Те же эмбеддинги и параметры БД, что в `ingest.py`. LLM для ответа: `Ollama` (`gpt-oss:20b`). Загрузка индекса из векторного стора: `VectorStoreIndex.from_vector_store`. Движок запросов: `as_query_engine(similarity_top_k=3)`, затем `query_engine.query(question)`. В конце выводятся источники (`source_nodes`: имя файла, score, фрагмент текста). |

**Запуск (из каталога, где лежит `content/` рядом со скриптами, как в репозитории для сдачи):**

```bash
python3 ingest.py
python3 query.py "Ваш вопрос по nginx"
```

**Зависимости:** LlamaIndex с интеграциями Postgres и Ollama (`llama-index-vector-stores-postgres`, `llama-index-embeddings-ollama`, `llama-index-llms-ollama` и ядро), доступный PostgreSQL с `pgvector`, запущенный Ollama с указанными моделями.

### Локальная копия документации и вспомогательные скрипты

| Путь / файл | Назначение |
|---------------|------------|
| [`practic/content/`](practic/content/) | Локальное дерево в духе репозитория nginx `content/` — чтобы можно было прогонять индексацию без клонирования GitHub на каждой машине. |
| [`practic/ingest_ollama.py`](practic/ingest_ollama.py), [`practic/query_ollama.py`](practic/query_ollama.py) | Вариант с **другими** именем БД и учётными данными (`myvectordb` / `user` / `secretpassword` в `ingest_ollama.py`) — удобно как черновик или под отдельный `docker-compose`; для проверки по ТЗ ориентируйтесь на `user-repo-bgrht` и БД `support`. |

**Соответствие критериям:** в `practic/user-repo-bgrht/` лежат эталонные для блока `ingest.py` и `query.py` на LlamaIndex с хранением векторов в PostgreSQL (pgvector) и полным циклом «индексация → поиск → ответ».
