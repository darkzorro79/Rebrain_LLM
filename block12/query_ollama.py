import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# import psycopg2

# 1. Глобальные настройки
OLLAMA_BASE_URL = "http://localhost:11434"

# Настраиваем эмбеддинги (для поиска в базе)
Settings.embed_model = OllamaEmbedding(
    model_name="qwen3-embedding:8b-q4_K_M",
    base_url=OLLAMA_BASE_URL,
)

# Настраиваем LLM (для генерации текста ответа)
Settings.llm = Ollama(
    model="gpt-oss:20b",
    base_url=OLLAMA_BASE_URL,
)

# 2. Настройка подключения к PostgreSQL
# Параметры из docker-compose.yml в директории base
DB_NAME = "myvectordb"
DB_USER = "user"
DB_PASSWORD = "secretpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

# 3. Инициализация того же Vector Store
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="rag_knowledge_base",  # Та же таблица!
    embed_dim=4096,  # Размерность вектора qwen3-embedding:8b-q4_K_M
)

# 4. Загрузка ИНДЕКСА из базы (без пересчета эмбеддингов)
# Обратите внимание: мы используем from_vector_store, а не from_documents
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 5. Создание Query Engine
# Это наш интерфейс "Вопрос -> Ответ"
query_engine = index.as_query_engine(similarity_top_k=3)  # Искать топ-3 куска

# 6. Выполнение запроса
print("🤖 Думаю...")
response = query_engine.query("Какой лимит на передачу данных в компании?")

# 7. Вывод ответа
print(f"\n💡 Ответ:\n{response}\n")

# 8. Проверка источников (Source Nodes)
print("-" * 30)
print("🔍 Использованные источники:")

for node in response.source_nodes:
    # node.score - насколько вектор похож (косинусное сходство)
    # node.metadata - имя файла, страница и т.д.
    print(f"📄 Файл: {node.metadata.get('file_name')} | Score: {node.score:.4f}")
    print(f"   Текст: {node.text[:100]}...")  # Первые 100 символов чанка
    print("-" * 10)