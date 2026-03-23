import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Инициализируем модель
# Используем gemini-2.5-flash для скорости и экономии
model = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    base_url="https://llm.rebrainme.study/",
    api_key="API_KEY"
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
    
    # Передаем значение для {topic}
    result = chain.invoke({"topic": "Docker"})
    
    print(f"Результат:\n{result}")
