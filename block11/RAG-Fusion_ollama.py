import os
from openai import OpenAI

# Инициализация клиента Ollama
client = OpenAI(
    api_key="ollama",  # Ollama не требует реального ключа, но клиент требует что-то указать
    base_url="http://localhost:11434/v1"  # Ollama OpenAI-compatible API endpoint
)

# Модели
LLM_MODEL = "gpt-oss:20b"


# --- Шаг 1: Генерация вариаций запроса ---

def generate_queries(original_query, n=4):
    """Генерирует n вариаций поискового запроса"""
    prompt = f"""
    Ты — AI-ассистент. Твоя задача — сгенерировать {n} разных поисковых запросов
    на основе вопроса пользователя, чтобы найти информацию в базе знаний с разных сторон.
    Вопрос: "{original_query}"
    Выведи только список запросов, каждый с новой строки.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    # Разбиваем ответ на список строк
    return response.choices[0].message.content.strip().split("\n")

# --- Шаг 2: Алгоритм RRF ---

def reciprocal_rank_fusion(results_list, k=60):
    """
    Принимает список списков результатов поиска.
    Возвращает один отсортированный список с новыми скорами.
    """
    fused_scores = {} # Словарь {doc_content: score}

    # Проходим по каждому списку результатов (от каждого запроса)
    for docs in results_list:
        # rank начинается с 0, поэтому rank+1
        for rank, doc in enumerate(docs):
            # Если документа еще нет в словаре, инициализируем
            if doc not in fused_scores:
                fused_scores[doc] = 0

            # Формула RRF: 1 / (константа + ранг)
            # Чем меньше ранг (выше в списке), тем больше число мы прибавляем
            fused_scores[doc] += 1 / (k + rank + 1)

    # Сортируем документы по итоговому скору (от большего к меньшему)
    # Возвращаем список кортежей (doc, score)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked_results

# --- Эмуляция работы ---

original_query = "Как исправить ошибку 502?"
generated_queries = generate_queries(original_query)

print(f"Оригинальный запрос: {original_query}")
print(f"Сгенерированные запросы: {generated_queries}\n")

# Представим, что мы сделали 3 поиска в Vector DB и получили такие результаты:
# (В реальности тут были бы вызовы vector_db.search())
search_results_1 = ["Док А: Проблема Nginx", "Док Б: Перезагрузка сервера", "Док В: Логи ошибок"]
search_results_2 = ["Док Г: Настройка PHP-FPM", "Док А: Проблема Nginx", "Док Д: Тайм-ауты"]
search_results_3 = ["Док А: Проблема Nginx", "Док Е: Балансировщик", "Док Г: Настройка PHP-FPM"]

# Объединяем списки в один
all_results = [search_results_1, search_results_2, search_results_3]

# Применяем RRF
final_ranking = reciprocal_rank_fusion(all_results)

print("--- Итоговый топ RAG-Fusion ---")
for rank, (doc, score) in enumerate(final_ranking):
    print(f"#{rank+1}: {doc} (RRF Score: {score:.4f})")