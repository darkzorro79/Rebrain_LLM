from openai import OpenAI
import numpy as np

# Если у вас есть GPU или Apple Silicon (M1/M2/M3), вы можете запустить
# локальный сервер (например, через llama.cpp или Ollama) и заменить
# base_url на http://localhost:8080/v1
client = OpenAI(
    api_key="API_KEY", # Для локального сервера можно вписать любой текст
    base_url="https://llm.rebrainme.study/" # Либо адрес вашего провайдера/локального хоста
)

def cos_sim(a, b):
    """Считает косинусное сходство между двумя векторами"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Наши данные
sentences = [
    "That is a happy dog",
    "That is a very happy person",
    "Today is a bad day"
]

# 2. Получаем эмбеддинги через API
# Используем модель gemini-embedding-001 (или указываем имя локальной модели)
response = client.embeddings.create(
    input=sentences,
    model="gemini-embedding-001"
)

# Извлекаем векторы из ответа
embeddings = [data.embedding for data in response.data]

# 3. Информация о векторах
print(f"Размерность вектора: {len(embeddings[0])}")
print(f"Вектор (первые 5 чисел): {embeddings[0][:5]} ...\n")

# 4. Считаем сходство
sim_dog_person = cos_sim(embeddings[0], embeddings[1])
sim_dog_weather = cos_sim(embeddings[0], embeddings[2])

print(f"Сходство: 'Happy Dog' vs 'Happy Person': {sim_dog_person:.4f}")
print(f"Сходство: 'Happy Dog' vs 'Bad Day':    {sim_dog_weather:.4f}")
