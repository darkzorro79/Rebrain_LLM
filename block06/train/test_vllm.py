#!/usr/bin/env python3
"""
Тестирование обученной пиратской модели через vLLM
vLLM обеспечивает быструю генерацию с использованием PagedAttention
Поддерживает работу как со слитой моделью, так и с LORA адаптерами
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Выберите один из вариантов:

# Вариант 1: Использование LORA адаптеров напрямую (быстрее, экономнее память)
USE_LORA = True
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/opt/train/qwen_final"

# Вариант 2: Использование слитой модели (если уже выполнили merge.py)
# USE_LORA = False
# MODEL_PATH = "/opt/train/qwen_merged_full"

print("="*60)
print("🏴‍☠️ Загрузка Пиратской Модели через vLLM")
print("="*60)

if USE_LORA:
    print(f"Базовая модель: {BASE_MODEL}")
    print(f"LORA адаптеры: {LORA_PATH}")
    print("="*60 + "\n")
    
    # Инициализация vLLM с LORA
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",              # BFloat16 для H100
        max_model_len=4096,            # Максимальная длина контекста
        gpu_memory_utilization=0.9,    # Использовать 90% VRAM
        trust_remote_code=True,        # Для Qwen моделей
        enable_lora=True,              # Включить поддержку LORA
    )
    
    # Загружаем LORA адаптеры
    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("pirate_lora", 1, LORA_PATH)
    
    print("✅ Модель и LORA адаптеры загружены!\n")
else:
    print(f"Модель: {MODEL_PATH}")
    print("="*60 + "\n")
    
    # Инициализация vLLM со слитой моделью
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",              # BFloat16 для H100
        max_model_len=4096,            # Максимальная длина контекста
        gpu_memory_utilization=0.9,    # Использовать 90% VRAM
        trust_remote_code=True,        # Для Qwen моделей
    )
    lora_request = None
    
    print("✅ Модель загружена!\n")

# Параметры генерации
sampling_params = SamplingParams(
    temperature=0.7,          # Творчество (0.0 = детерминизм, 1.0+ = креатив)
    top_p=0.9,               # Nucleus sampling
    max_tokens=256,          # Максимум токенов в ответе
    repetition_penalty=1.1,  # Штраф за повторы
)

# Загружаем токенизатор для форматирования промптов
if USE_LORA:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Тестовые запросы
test_prompts = [
    "Как дела?",
    "Расскажи мне о себе",
    "Что такое машинное обучение?",
    "Как приготовить борщ?",
]

print("🏴‍☠️ ТЕСТИРОВАНИЕ ПИРАТСКОЙ МОДЕЛИ")
print("="*60)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n📝 Запрос {i}: {prompt}")
    print("-"*60)
    
    # Формируем сообщение в формате чата
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Применяем chat template (Qwen формат)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Генерация
    if USE_LORA:
        outputs = llm.generate([formatted_prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([formatted_prompt], sampling_params)
    
    # Вывод результата
    generated_text = outputs[0].outputs[0].text
    print(f"🏴‍☠️ Ответ: {generated_text}")
    print("-"*60)

print("\n" + "="*60)
print("✅ Тестирование завершено!")
print("="*60)

# Интерактивный режим
print("\n💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("Введите 'exit' или 'quit' для выхода\n")

while True:
    try:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("👋 До свидания!")
            break
        
        if not user_input:
            continue
        
        messages = [{"role": "user", "content": user_input}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        if USE_LORA:
            outputs = llm.generate([formatted_prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([formatted_prompt], sampling_params)
        
        response = outputs[0].outputs[0].text
        print(f"🏴‍☠️ Модель: {response}\n")
        
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
        break
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
