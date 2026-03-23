import os
# Импортируем ключевые компоненты
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import psycopg2

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
# Мы указываем таблицу, размерность вектора и параметры подключения
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="rag_knowledge_base",  # Имя таблицы, куда упадут векторы
    embed_dim=4096,  # ВАЖНО: Размерность вектора qwen3-embedding:8b-q4_K_M
)

# 4. Создание контекста хранилища (Storage Context)
# Это "клей", который говорит LlamaIndex: "Храни векторы не в памяти, а вот в этом объекте vector_store"
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Загрузка данных (ETL - Extract)
# SimpleDirectoryReader рекурсивно читает папку и создает объекты Document
print("🚀 Загружаем документы...")
documents = SimpleDirectoryReader("./data").load_data()
print(f"📄 Загружено документов: {len(documents)}")

# 6. Создание индекса (ETL - Transform & Load)
# Самая важная строка. Здесь происходит вся магия.
print("⚙️ Создаем индекс и сохраняем в Postgres...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True  # Показывать прогресс-бар
)

print("✅ Индексация завершена! Данные в Postgres.")