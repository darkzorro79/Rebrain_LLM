from sentence_transformers import SentenceTransformer, util

# 1. Загружаем модель
# При первом запуске она скачается автоматически (~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Наши данные
sentences = [
    "That is a happy dog",       # Предложение 0: Живое существо, эмоция радости
    "That is a very happy person", # Предложение 1: Живое существо, эмоция радости
    "Today is a bad day"       # Предложение 2: Погода, негатив, другой контекст
]

# 3. Превращаем текст в векторы (Encoding)
embeddings = model.encode(sentences)

# Посмотрим на размерность и сам вектор
print(f"Размерность вектора: {len(embeddings[0])}")
print(f"Вектор (первые 5 чисел): {embeddings[0][:5]} ...\n")

# 4. Считаем косинусное сходство (Cosine Similarity)
# Сравниваем "Собаку" (0) с "Человеком" (1)
sim_dog_person = util.cos_sim(embeddings[0], embeddings[1])

# Сравниваем "Собаку" (0) с "Солнечным днём" (2)
sim_dog_weather = util.cos_sim(embeddings[0], embeddings[2])

print(f"Сходство: 'Happy Dog' vs 'Happy Person': {sim_dog_person.item():.4f}")
print(f"Сходство: 'Happy Dog' vs 'Bad Day':    {sim_dog_weather.item():.4f}")
