from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Инициализируем модель Ollama
model = ChatOllama(
    model="qwen3.5:35b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 2. Создаем шаблон промпта. Переменная {topic} будет подставлена позже
prompt = ChatPromptTemplate.from_template(
    "Объясни концепцию {topic} в одном предложении, как будто мне 5 лет."
)

# 3. Инициализируем парсер (он превратит объект AIMessage в обычную строку)
parser = StrOutputParser()

# 4. СБОРКА ПАЙПЛАЙНА С ПОМОЩЬЮ LCEL
# Читается как: "Возьми переменные -> вставь в промпт -> отправь в модель -> распарси в строку"
chain = prompt | model | parser

# 5. Запускаем цепочку методом invoke(), передавая словарь с переменными
if __name__ == "__main__":
    print("Запускаем генерацию...\n")
    
    # Streaming: получаем ответ по частям в реальном времени
    for chunk in chain.stream({"topic": "Docker"}):
        print(chunk, end="", flush=True)
    
    print("\n")