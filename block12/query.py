import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
# import psycopg2

# 1. Глобальные настройки
API_KEY = "API_KEY"
BASE_URL = "https://llm.rebrainme.study"

# Настраиваем эмбеддинги (для поиска в базе)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=API_KEY,
    api_base=BASE_URL,
)

# Настраиваем LLM (для генерации текста ответа)
Settings.llm = OpenAI(
    model="gpt-5.2",
    api_key=API_KEY,
    api_base=BASE_URL,
)

# 2. Настройка подключения к PostgreSQL
# Параметры должны совпадать с теми, что мы указывали при создании базы
DB_NAME = "app"
DB_USER = "app"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"

# 3. Инициализация того же Vector Store
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="rag_knowledge_base", # Та же таблица!
    embed_dim=1536,
)

# 4. Загрузка ИНДЕКСА из базы (без пересчета эмбеддингов)
# Обратите внимание: мы используем from_vector_store, а не from_documents
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 5. Создание Query Engine
# Это наш интерфейс "Вопрос -> Ответ"
query_engine = index.as_query_engine(similarity_top_k=3) # Искать топ-3 куска

# 6. Выполнение запроса
print("🤖 Думаю...")
response = query_engine.query("Какой лимит на передачу данных в компании?")

# 6. Вывод ответа
print(f"\n💡 Ответ:\n{response}\n")
