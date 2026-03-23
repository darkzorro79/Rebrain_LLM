import psycopg2
from openai import OpenAI

# 1. Настройка клиента для эмбеддингов (Ollama)
client_ai = OpenAI(
    # Ollama не требует API‑ключа, оставляем пустым
    api_key="",
    base_url="http://localhost:11434/v1"
)

# 2. Подключение к базе PostgreSQL (контейнер pgvector-db)
conn = psycopg2.connect(
    dbname="myvectordb",
    user="user",
    password="secretpassword",
    host="localhost"
)
cur = conn.cursor()

# Создаём таблицу (4096 — размерность вектора для модели qwen3‑embedding:8b‑q4_K_M)
cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id serial PRIMARY KEY,
        content text,
        embedding vector(4096),
        source text
    );
""")
conn.commit()

# 3. Подготовка данных
documents = [
    "Квантовый процессор Rebrain (RQP) требует температуру охлаждения строго -273.14 градусов Цельсия для работы.",
    "В отличие от обычных квантовых компьютеров на кубитах, архитектура RQP основана на 'Re-битах', которые могут находиться в трёх состояниях одновременно.",
    "Критическая ошибка E-99 в системах RQP означает, что оператор забыл покормить кота Шрёдингера. Требуется немедленное вмешательство."
]
sources = ["manual_v1.pdf", "whitepaper.txt", "error_logs.json"]

def get_embedding(text):
    """Получить эмбеддинг текста через Ollama.

    Используем модель ``qwen3-embedding:8b-q4_K_M`` – она выдаёт вектор
    размерностью 4096, что совпадает с типом столбца ``vector(4096)``.
    """
    response = client_ai.embeddings.create(
        input=[text],
        model="qwen3-embedding:8b-q4_K_M"
    )
    # ``response.data`` – список объектов, каждый имеет поле ``embedding``
    return response.data[0].embedding

# 4. Индексация
print("Начинаю генерацию эмбеддингов и сохранение в Postgres...")
for doc, src in zip(documents, sources):
    embedding = get_embedding(doc)
    cur.execute(
        "INSERT INTO docs (content, embedding, source) VALUES (%s, %s, %s)",
        (doc, embedding, src)
    )

conn.commit()
cur.close()
conn.close()
print("Все документы успешно проиндексированы!")
