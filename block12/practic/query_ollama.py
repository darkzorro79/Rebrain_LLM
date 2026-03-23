import os
import sys
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 1. Проверка аргументов командной строки
if len(sys.argv) < 2:
    print("❌ Использование: python3 query_ollama.py <вопрос>")
    print("   Пример: python3 query_ollama.py \"Как настроить nginx?\"")
    sys.exit(1)

question = sys.argv[1]

# 2. Глобальные настройки
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

# 3. Настройка подключения к PostgreSQL
# Параметры из docker-compose.yml в директории base
DB_NAME = "myvectordb"
DB_USER = "user"
DB_PASSWORD = "secretpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

# 4. Инициализация Vector Store
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name="rag_nginx_knowledge_base",  # Таблица с документацией nginx
    embed_dim=4096,  # Размерность вектора qwen3-embedding:8b-q4_K_M
)

# 5. Загрузка индекса из базы
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 6. Создание Query Engine
query_engine = index.as_query_engine(similarity_top_k=3)

# 7. Выполнение запроса
print(f"🤖 Вопрос: {question}")
print("💭 Думаю...")
response = query_engine.query(question)

# 8. Вывод ответа
print(f"\n💡 Ответ:\n{response}\n")

# 9. Проверка источников (Source Nodes)
print("-" * 30)
print("🔍 Использованные источники:")

for node in response.source_nodes:
    print(f"📄 Файл: {node.metadata.get('file_name')} | Score: {node.score:.4f}")
    print(f"   Текст: {node.text[:100]}...")
    print("-" * 10)