#!/usr/bin/env python3
"""
Скрипт для слияния базовой модели с адаптерами LORA
Создает полную модель, которую можно использовать напрямую или через vLLM
"""

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Настройка прокси (если нужно скачивать базовую модель)
# Раскомментируйте при необходимости
# os.environ['HTTP_PROXY'] = 'socks5://103.137.249.145:3001'
# os.environ['HTTPS_PROXY'] = 'socks5://103.137.249.145:3001'

# 1. Пути
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"     # Базовая модель
ADAPTER_PATH = "/opt/train/qwen_final"           # Путь к LORA адаптерам
OUTPUT_PATH = "/opt/train/qwen_merged_full"      # Куда сохранить готовую модель

print("="*60)
print("🔧 Слияние модели с адаптерами LORA")
print("="*60)
print(f"Базовая модель: {BASE_MODEL_NAME}")
print(f"LORA адаптеры: {ADAPTER_PATH}")
print(f"Выходная папка: {OUTPUT_PATH}")
print("="*60 + "\n")

# 2. Загрузка базовой модели
print(f"Загружаю базовую модель: {BASE_MODEL_NAME}...")

# Для H100 96GB можем загрузить модель в BF16 без проблем
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    return_dict=True,
    torch_dtype=torch.bfloat16,  # BFloat16 оптимально для H100
    device_map="auto",           # Автоматически на GPU
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("✅ Модель загружена в память\n")

# 3. Загрузка адаптеров LORA
print(f"Загружаю адаптеры из {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("✅ Адаптеры загружены\n")

# 4. Слияние весов
print("Начинаю процесс слияния (merge_and_unload)...")
print("Это математически складывает веса LoRA с базовыми весами модели")
model = model.merge_and_unload()
print("✅ Слияние завершено!\n")

# 5. Сохранение объединенной модели
print(f"Сохраняю объединенную модель в {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
print("✅ Веса модели сохранены\n")

# 6. Сохранение токенизатора
print("Сохраняю токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print("✅ Токенизатор сохранен\n")

# Итоговое сообщение
print("="*60)
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
print("="*60 + "\n")
