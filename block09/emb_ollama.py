from openai import OpenAI
import numpy as np

# Ollama runs locally on port 11434. Use the OpenAI compatible endpoint.
# Any api_key works for local Ollama; a placeholder is fine.
client = OpenAI(
    api_key="dummy",  # placeholder for local Ollama
    base_url="http://localhost:11434/v1"
)

def cos_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Sample sentences
sentences = [
    "That is a happy dog",
    "That is a very happy person",
    "Today is a bad day"
]

# Get embeddings from Ollama model qwen3-embedding:8b-q4_K_M
response = client.embeddings.create(
    input=sentences,
    model="qwen3-embedding:8b-q4_K_M"
)

# Extract vectors
embeddings = [data.embedding for data in response.data]

# Show vector info
print(f"Vector dimension: {len(embeddings[0])}")
print(f"Vector (first 5 values): {embeddings[0][:5]} ...\n")

# Compute similarities
sim_dog_person = cos_sim(embeddings[0], embeddings[1])
sim_dog_weather = cos_sim(embeddings[0], embeddings[2])

print(f"Similarity 'Happy Dog' vs 'Happy Person': {sim_dog_person:.4f}")
print(f"Similarity 'Happy Dog' vs 'Bad Day':    {sim_dog_weather:.4f}")
