from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. Читаем файл с политикой
with open('policy.md', 'r', encoding='utf-8') as file:
    content = file.read()

headers_to_split_on = [
    ("#", "Department"),
    ("##", "Topic"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(content)

print(f"\n\n=== Markdown Splitter ({len(md_chunks)} чанков) ===\n")
for chunk in md_chunks:
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}") # Самое важное здесь!
    print("-" * 10)
