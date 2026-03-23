import psycopg2
from openai import OpenAI

# 1. Настройка клиента Ollama
client_ai = OpenAI(
    api_key="",                # Ollama не требует API‑ключа
    base_url="http://localhost:11434/v1"
)

# 2. Подключение к PostgreSQL (контейнер pgvector-db)
conn = psycopg2.connect(
    dbname="myvectordb",
    user="user",
    password="secretpassword",
    host="localhost"
)
cur = conn.cursor()

# 3. Текстовый запрос пользователя
query_text = "Почему система сломалась из‑за домашнего животного?"
print(f"Запрос пользователя: '{query_text}'")

# 4. Преобразуем запрос в эмбеддинг через Ollama
query_embedding = client_ai.embeddings.create(
    input=[query_text],
    model="qwen3-embedding:8b-q4_K_M"
).data[0].embedding

# 5. Поиск ближайшего вектора в базе (L2‑расстояние)
cur.execute(
    """
    SELECT content, source, embedding <=> %s::vector AS distance
    FROM docs
    ORDER BY distance ASC
    LIMIT 1;
    """,
    (query_embedding,)
)

result = cur.fetchone()

print("-" * 50)
if result:
    # result = (content, source, distance)
    print(f"ЛУЧШИЙ РЕЗУЛЬТАТ (дистанция: {result[2]:.4f}):")
    print(f"Текст: {result[0]}")
    print(f"Источник: {result[1]}")
else:
    print("Ничего не найдено – база пуста или запрос не дал совпадений.")

cur.close()
conn.close()
