# Система семантического поиска по Kubernetes CHANGELOG v1.35

Эта система реализует семантический поиск по списку изменений Kubernetes v1.35 с использованием векторной базы данных PostgreSQL и эмбеддингов Gemini.

## Требования

- Python 3.9+
- PostgreSQL с расширением pgvector
- API key для доступа к OpenAI-совместимому API (для эмбеддингов и LLM)

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Настройка PostgreSQL

1. Установите PostgreSQL и pgvector:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
```

2. Установите pgvector:
```bash
# Следуйте инструкциям на https://github.com/pgvector/pgvector
```

3. Создайте базу данных:
```sql
CREATE DATABASE rag_db;
\c rag_db;
CREATE EXTENSION vector;

CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(3072)
);
```

## Настройка окружения

Скопируйте `.env.example` в `.env` и заполните переменные:

```bash
cp .env.example .env
# Отредактируйте .env файл
```

Или экспортируйте переменные напрямую:
```bash
export OPENAI_API_KEY="sk-pJZPSWNCDe3vYu0tXj0pnw"
export OPENAI_BASE_URL="https://llm.rebrainme.study"
export DB_DSN="postgresql://user:password@localhost:5432/rag_db"
```

## Использование

### 1. Индексация данных

Запустите скрипт индексации для загрузки и обработки CHANGELOG:

```bash
python3 indexer.py
```

Скрипт:
- Загрузит CHANGELOG-1.35.md с GitHub
- Разобьет на чанки с помощью MarkdownTextSplitter
- Сгенерирует эмбеддинги с помощью gemini-embedding-001
- Сохранит в базу данных

### 2. Поиск и ответы

Задайте вопрос системе:

```bash
python3 rag.py "Что нового в работе с API?"
python3 rag.py "Какие изменения в SIG Instrumentation?"
```

Скрипт:
- Сгенерирует эмбеддинг для вопроса
- Найдет 3 наиболее релевантных чанка
- Сгенерирует ответ с помощью gemini-2.5-flash

## Архитектура

### Файловая структура:
- **indexer.py**: Скрипт индексации changelog
- **rag.py**: Скрипт для поиска и генерации ответов
- **requirements.txt**: Зависимости Python
- **PostgreSQL**: База данных с pgvector для хранения эмбеддингов

### Техническая архитектура:
- **Эмбеддинги**: OpenAI Protocol → `gemini-embedding-001` (3072 dim) @ llm.rebrainme.study
- **Ответы**: OpenAI Protocol → `gemini-2.5-flash` @ llm.rebrainme.study
- **БД**: PostgreSQL + pgvector через `DB_DSN`
- **Поиск**: Векторное сходство через операторы `<=>` или `<->`

## Модели

- **Эмбеддинги**: `gemini-embedding-001` (3072 измерения) через OpenAI-совместимый API
- **Генерация ответов**: `gemini-2.5-flash` через OpenAI-совместимый API

## Устранение неполадок

1. Убедитесь, что OPENAI_API_KEY и OPENAI_BASE_URL корректны
2. Проверьте, что модель `gemini-embedding-001` доступна через API endpoint
3. Проверьте подключение к PostgreSQL через DB_DSN
4. Убедитесь, что расширение pgvector установлено
5. Проверьте, что таблица knowledge_base создана с правильной схемой