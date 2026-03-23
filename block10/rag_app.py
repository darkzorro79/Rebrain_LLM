import os
import psycopg2
from openai import OpenAI

# 1. Настройка клиентов
client = OpenAI(
    api_key="sk-gTGAwOGHKdq0QhxbYIZ9yQ", # Для локального сервера можно вписать любой текст
    base_url="https://llm.rebrainme.study/" # Либо адрес вашего провайдера/локального хоста
)

# 2. Подключение к базе
conn = psycopg2.connect(
    dbname="app",
    user="app",
    password="password",
    host="localhost"
)
cur = conn.cursor()

def get_embedding(text: str) -> list:
    """Генерация вектора вопроса"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model="gemini-embedding-001"
    )
    return response.data[0].embedding

def search_db(query_vector: list, k: int = 3):
    """Поиск контекста в PostgreSQL"""
    # Ищем похожие векторы
    sql = """
        SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """
    cur.execute(sql, (query_vector, query_vector, k))
    results = cur.fetchall()

    # Упаковываем в удобный формат
    return [{"content": r[0], "metadata": r[1], "score": r[2]} for r in results]

def ask_rag(question: str):
    """
    Главная функция-оркестратор.
    """
    print(f"\n🤖 Пользователь: {question}")

    # Шаг 1: Векторизация вопроса
    print("   ... Векторизую вопрос")
    q_vector = get_embedding(question)

    # Шаг 2: Поиск (Retrieval)
    print("   ... Ищу информацию в базе знаний")
    context_items = search_db(q_vector, k=3)

    if not context_items:
        return "База знаний пуста или недоступна."

    # Логирование найденного (для отладки)
    print(f"   ... Найдено {len(context_items)} фрагментов. Топ скор: {context_items[0]['score']:.4f}")

    # Шаг 3: Сборка промпта (Augmentation)
    context_str = "\n\n---\n\n".join([item['content'] for item in context_items])

    system_prompt = """
    Ты — корпоративный AI-ассистент. Отвечай на вопросы, используя ТОЛЬКО предоставленный ниже контекст.
    Если информации в контексте недостаточно, ответь: "Я не нашёл информации в базе знаний".
    Не придумывай факты. Ссылайся на разделы документов, если это возможно.
    """

    user_prompt = f"""
    CONTEXT:
    {context_str}

    QUESTION:
    {question}
    """

    # Шаг 4: Генерация (Generation)
    print("   ... Генерирую ответ")
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1 # Минимальная креативность для точности
    )

    return response.choices[0].message.content

# --- Запуск сценариев ---
if __name__ == "__main__":
    # Сценарий 1: Вопрос, ответ на который есть в документах (про пароли из примера)
    answer1 = ask_rag("Какие требования к длине пароля?")
    print(f"\n💬 Ответ RAG:\n{answer1}\n")
    print("="*50)

    # Сценарий 2: Вопрос, ответа на который НЕТ в документах
    answer2 = ask_rag("Как приготовить лазанью?")
    print(f"\n💬 Ответ RAG:\n{answer2}\n")
