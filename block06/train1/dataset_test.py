from datasets import load_dataset

# Загружаем созданный нами файл
dataset = load_dataset("json", data_files="pirate_dataset.jsonl", split="train")

# Проверяем, что загрузилось
print(dataset[0])

