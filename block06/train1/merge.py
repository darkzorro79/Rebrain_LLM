import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Настройка прокси (если нужно скачивать базовую модель)
# Раскомментируйте, если модель еще не скачана
# os.environ['HTTP_PROXY'] = 'socks5://103.137.249.145:3001'
# os.environ['HTTPS_PROXY'] = 'socks5://103.137.249.145:3001'

# 1. Пути
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Та же, что была при обучении
ADAPTER_PATH = "./qwen_pirate_final"          # Путь к сохраненным адаптерам
OUTPUT_PATH = "./qwen_pirate_merged_full"     # Куда сохранить готовую модель

print(f"Загружаю базовую модель: {BASE_MODEL_NAME}")

# ВАЖНО: Для слияния загружаем модель в BF16 (не 4-bit!)
# RTX 5090 имеет 32GB VRAM, модель 7B в BF16 занимает ~14GB - должно влезть
# Если не влезет, измените device_map на "cpu" (но это будет медленнее)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    return_dict=True,
    dtype=torch.bfloat16,      # BFloat16 оптимально для RTX 5090
    device_map="auto",         # Автоматически на GPU, fallback на CPU если не влезет
    low_cpu_mem_usage=True,
)

print("Модель загружена в память")

print(f"Загружаю адаптеры из {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("Адаптеры загружены")

print("\nНачинаю процесс слияния (merge_and_unload)...")
print("Это математически складывает веса LoRA с базовыми весами модели")
# Это ключевая команда: она математически складывает веса LoRA с базовыми весами
model = model.merge_and_unload()
print("Слияние завершено!")

print(f"\nСохраняю объединенную модель в {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True) # safe_serialization=True сохранит в safetensors
print("Веса модели сохранены")

# Не забываем про токенизатор! Его тоже надо скопировать в новую папку.
print("Сохраняю токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n" + "="*60)
print("✅ ГОТОВО! Пиратская модель успешно создана!")
print("="*60)
print(f"\nПуть к модели: {OUTPUT_PATH}")
print("\nТеперь можно:")
print("1. Использовать модель с transformers")
print("2. Запустить через vLLM для быстрого инференса")
print("3. Загрузить на HuggingFace Hub")
print("\nПример использования:")
print(f'  model = AutoModelForCausalLM.from_pretrained("{OUTPUT_PATH}")')
print(f'  tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_PATH}")')
