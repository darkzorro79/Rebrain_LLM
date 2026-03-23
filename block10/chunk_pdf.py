import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """
    Очистка сырого текста из PDF.
    """
    # 1. Удаляем номера страниц (например, "Page 1 of 10" или просто цифры внизу)
    # Это простая регулярка, под реальный документ может потребоваться сложнее
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

    # 2. Склеиваем переносы слов (hyphenation): "инфор-\nмация" -> "информация"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 3. Удаляем множественные пробелы и лишние пустые строки
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_pdf(file_path: str):
    """
    Загружает PDF, чистит текст и режет на чанки.
    """
    print(f"--- Обработка файла: {file_path} ---")

    # 1. Загрузка (Lazy Load не нужен, грузим всё в память)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Загружено страниц: {len(documents)}")

    # 2. Очистка текста на каждой странице
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        # Полезно добавить метаданные источника, если их нет
        doc.metadata['source'] = file_path

    # 3. Нарезка (Chunking)
    # Используем размер побольше, чем в демо (1000 символов ~ 200-300 токенов)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # split_documents автоматически пробрасывает metadata (номер страницы) в чанки!
    chunks = text_splitter.split_documents(documents)

    print(f"Сформировано чанков: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    # Создайте или скачайте любой PDF для теста
    try:
        chunks = process_pdf("policy.pdf")

        # Посмотрим на первый чанк
        if chunks:
            print("\n--- Пример первого чанка ---")
            print(f"Content: {chunks[0].page_content[:200]}...") # Первые 200 символов
            print(f"Metadata: {chunks[0].metadata}")
    except Exception as e:
        print(f"Ошибка: {e}. Пожалуйста, положите файл policy.pdf в папку со скриптом.")
