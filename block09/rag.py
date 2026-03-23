import psycopg2
from openai import OpenAI

# 1️⃣ Клиент Ollama для эмбеддингов (модель qwen3‑embedding:8b‑q4_K_M)
emb_client = OpenAI(
    api_key="",                     # Ollama не требует API‑ключа
    base_url="http://localhost:11434/v1"
)

# 2️⃣ Клиент для генерации ответа (модель gpt‑oss:120b‑cloud)
# Предполагаем, что модель доступна через тот же endpoint Ollama или иной OpenAI‑совместимый сервис.
# Если у вас отдельный URL – замените его ниже.
gen_client = OpenAI(
    api_key="",                     # тоже без ключа, если локальный сервис
    base_url="http://localhost:11434/v1"
)

def get_context(question: str) -> str:
    """Извлекает наиболее похожий фрагмент текста из базы PostgreSQL.

    1️⃣ Получаем эмбеддинг вопроса через Ollama.
    2️⃣ Находим документ с минимальным L2‑расстоянием (оператор <=>).
    """
    # Подключаемся к базе, созданной в docker‑compose
    conn = psycopg2.connect(
        dbname="myvectordb",
        user="user",
        password="secretpassword",
        host="localhost"
    )
    cur = conn.cursor()

    # 1️⃣ Эмбеддинг вопроса
    emb = emb_client.embeddings.create(
        input=[question],
        model="qwen3-embedding:8b-q4_K_M"
    ).data[0].embedding

    # 2️⃣ Поиск самого релевантного документа
    cur.execute(
        "SELECT content FROM docs ORDER BY embedding <=> %s::vector LIMIT 1",
        (emb,)
    )
    row = cur.fetchone()
    context = row[0] if row else ""

    cur.close()
    conn.close()
    return context

def generate_rag_answer(question: str):
    print(f"\n🤖 ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}")

    # ШАГ 1: Поиск контекста (Retrieval)
    context = get_context(question)
    if not context:
        print("⚠️ Не найдено подходящего контекста в базе.")
        return
    print(f"🔎 НАЙДЕННЫЙ КОНТЕКСТ В БД: {context}")

    # ШАГ 2: Формирование промпта (Augmentation)
    system_prompt = f"""
Ты — ведущий инженер R&D отдела.
Используй ТОЛЬКО предоставленный контекст для ответа на вопрос.
Если в контексте нет нужной информации, ответь: 'К сожалению, в базе знаний нет данных для ответа'.

КОНТЕКСТ:
{context}
"""

    # ШАГ 3: Генерация ответа (Generation) через модель gpt‑oss
    response = gen_client.chat.completions.create(
        model="gpt-oss:120b-cloud",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    print("-" * 40)
    print(f"✅ ОТВЕТ LLM:\n{response.choices[0].message.content}")

if __name__ == "__main__":
    # Пример вопроса, который заставит модель искать данные в базе
    generate_rag_answer("Какая температура нужна для работы процессора RQP?")

