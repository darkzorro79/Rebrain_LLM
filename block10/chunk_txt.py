from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Читаем файл с политикой
with open('policy.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    
# 2. Настройка сплиттера
# chunk_size: Максимальный размер одного куска (в символах).
# chunk_overlap: Количество символов, которые дублируются в конце предыдущего и начале следующего чанка.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # 20-30% от размера чанка — хорошая практика
    length_function=len,
    separators=["\n\n", "\n", " ", ""] # Порядок приоритета разделителей
)

# 3. Нарезка текста
chunks = text_splitter.create_documents([content])

# 4. Вывод результатов
print(f"Исходная длина текста: {len(content)} символов")
print(f"Количество чанков: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- Чанк {i+1} (Длина: {len(chunk.page_content)}) ---")
    print(chunk.page_content)
    print("-" * 20)
