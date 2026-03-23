from sentence_transformers import CrossEncoder

# --- Подготовка данных ---

query = "Почему не удаётся подключиться к базе данных?"

documents = [
    "Док 1: «Отдел маркетинга вчера обсуждал стратегию провальных подключений к базам данных».",
    "Док 2: «Для исправления ошибок подключения проверьте параметр max_connections в postgresql.conf и перезапустите службу».",
    "Док 3: «Строки подключения к базе данных должны храниться в секретах»."
]

# --- Логика переранжирования ---

# Инициализация модели Cross-Encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Формирование пар "Запрос — Документ"
pairs = [(query, doc) for doc in documents]

# Вычисление оценок релевантности
scores = model.predict(pairs)

# Сортировка документов по убыванию балла
ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# --- Вывод результатов ---

print(f"Запрос: {query}\n")
print("--- Результаты переранжирования ---")
for rank, (doc, score) in enumerate(ranked_results):
    print(f"#{rank+1}: {doc}")
    print(f"   Score: {score:.4f}\n")