#!/usr/bin/env python3
"""
Тестирование обученной пиратской модели через vLLM
vLLM обеспечивает быструю генерацию с использованием PagedAttention
"""

from vllm import LLM, SamplingParams

# Путь к вашей смерженной модели
MODEL_PATH = "./qwen_pirate_merged_full"

print("="*60)
print("🏴‍☠️ Загрузка Пиратской Модели через vLLM")
print("="*60)

# Инициализация vLLM
# dtype="bfloat16" оптимально для RTX 5090
# max_model_len можно увеличить до 8192 или больше
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",           # BFloat16 для RTX 5090
    max_model_len=4096,         # Максимальная длина контекста
    gpu_memory_utilization=0.9, # Использовать 90% VRAM
    trust_remote_code=True,     # Для Qwen моделей
)

print("\n✅ Модель загружена!\n")

# Параметры генерации
sampling_params = SamplingParams(
    temperature=0.7,      # Творчество (0.0 = детерминизм, 1.0+ = креатив)
    top_p=0.9,           # Nucleus sampling
    max_tokens=256,      # Максимум токенов в ответе
    repetition_penalty=1.1,  # Штраф за повторы
)

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
    # Нужно получить токенизатор из модели
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Генерация
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

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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
        
        outputs = llm.generate([formatted_prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        print(f"🏴‍☠️ Модель: {response}\n")
        
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
        break
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
