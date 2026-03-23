import os
import sys
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import psycopg2

# Отключаем логи HTTP запросов для чистого прогресс-бара
logging.getLogger("httpx").setLevel(logging.WARNING)

# 1. Глобальная настройка модели эмбеддингов
# Используем Ollama с моделью qwen3-embedding:8b-q4_K_M
Settings.embed_model = OllamaEmbedding(
    model_name="qwen3-embedding:8b-q4_K_M",
    base_url="http://localhost:11434",
)

# 2. Настройка подключения к PostgreSQL
# Параметры из docker-compose.yml в директории base
DB_NAME = "myvectordb"
DB_USER = "user"
DB_PASSWORD = "secretpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

# 3. Инициализация векторного хранилища (Vector Store)
# Используем новую таблицу rag_nginx_knowledge_base
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="rag_nginx_knowledge_base",  # Новая таблица для nginx документации
    embed_dim=4096,  # Размерность вектора qwen3-embedding:8b-q4_K_M
)

# 4. Создание контекста хранилища (Storage Context)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Загрузка данных (ETL - Extract)
# SimpleDirectoryReader рекурсивно читает папку content и создает объекты Document
print("🚀 Загружаем документы из content/...")
raw_documents = SimpleDirectoryReader("./content", recursive=True).load_data()
print(f"📄 Загружено документов: {len(raw_documents)}")

# 5.1 Очистка документов от NUL символов (0x00) — PostgreSQL их не принимает
print("🧹 Очистка документов от NUL символов...")
documents = []
for doc in raw_documents:
    cleaned_text = doc.text.replace('\x00', '') if doc.text else ""
    cleaned_doc = Document(text=cleaned_text, metadata=doc.metadata)
    documents.append(cleaned_doc)
print(f"✅ Очистка завершена, документов: {len(documents)}")

# 6. Создание индекса (ETL - Transform & Load)
print("⚙️ Создаем индекс и сохраняем в Postgres...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True  # Показывать прогресс-бар
)

print("✅ Индексация завершена! Данные в Postgres.")