from sentence_transformers import CrossEncoder

# 1. Загружаем модель Cross-Encoder
# Эта модель обучена на датасете MS MARCO (вопрос-ответ)
# Она маленькая и быстрая, подходит для CPU/GPU инференса в реальном времени
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Имитация данных
user_query = "Как оформить возврат товара?"

# Представим, что векторная база данных (этап 1) вернула нам эти 4 документа.
# Обратите внимание: некоторые документы содержат ключевые слова, но не отвечают на вопрос.
retrieved_documents = [
    "Политика компании запрещает возврат товара после 30 дней использования.", # Релевантно (частично)
    "Для оформления возврата перейдите в личный кабинет и нажмите кнопку 'Вернуть'.", # Идеальный ответ
    "Возврат инвестиций (ROI) в этом квартале составил 15%.", # Ловушка: слово возврат есть, смысл другой.
    "На складе заканчиваются товары категории А." # Шум
]

# 3. Подготовка пар для оценки
# Cross-Encoder требует список пар: [[Query, Doc1], [Query, Doc2], ...]
pairs = [[user_query, doc] for doc in retrieved_documents]

# 4. Получение скоров (Scores)
# Модель выдаёт число (логит); чем больше, тем релевантнее.
scores = cross_encoder.predict(pairs)

# 5. Объединение и сортировка
results = []
for doc, score in zip(retrieved_documents, scores):
    results.append({"doc": doc, "score": score})

# Сортируем по убыванию скора
ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

# 6. Вывод топ-3
print(f"Запрос: {user_query}\n")
print("--- Результаты после Re-ranking ---")
for i, res in enumerate(ranked_results[:3]):
    print(f"Rank {i+1} (Score: {res['score']:.4f}): {res['doc']}")
