import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig

# Настройка прокси для загрузки моделей (если требуется)
# Раскомментируйте при необходимости:
# os.environ['HTTP_PROXY'] = 'socks5://103.137.249.145:3001'
# os.environ['HTTPS_PROXY'] = 'socks5://103.137.249.145:3001'

# 1. Настройки модели
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 2. Загрузка модели БЕЗ квантизации (H100 96GB позволяет)
# На H100 с 96GB VRAM модель 7B в BF16 (~14GB) + градиенты + оптимизатор
# легко помещаются в память
print(f"Загрузка модели {MODEL_NAME}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,    # BFloat16 для оптимальной производительности
    device_map="auto",             # Автоматическое распределение по GPU
    trust_remote_code=True,        # Для Qwen моделей
)

# Включаем gradient checkpointing для дополнительной экономии памяти
model.gradient_checkpointing_enable()

print("Модель загружена!")

# 3. Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "right"  # Для SFTTrainer с packing=False

# Проверка pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Конфигурация LoRA 
# Увеличиваем параметры для лучшего качества (есть память)
peft_config = LoraConfig(
    r=64,                    # Увеличенный ранг для H100 (было 32)
    lora_alpha=128,          # Alpha = r * 2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"  # Обучаем все линейные слои
)

# 5. Загрузка датасета
dataset = load_dataset("json", data_files="pirate_dataset.jsonl", split="train")

# Функция для форматирования промптов
def formatting_prompts_func(examples):
    """Применяет chat template к сообщениям"""
    convos = examples["messages"]
    
    # Проверка типа: если пришел один пример, оборачиваем в список
    if isinstance(convos, list) and len(convos) > 0 and isinstance(convos[0], dict):
        convos = [convos]
    
    texts = []
    for messages in convos:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    return texts

# 6. Параметры обучения (оптимизированы для H100 96GB)
training_args = TrainingArguments(
    output_dir="/opt/train/qwen_pirate_v1",      # Директория для чекпоинтов
    num_train_epochs=3,                           
    per_device_train_batch_size=8,                # Увеличен для H100
    gradient_accumulation_steps=2,                # Эффективный batch = 8*2 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,                                    # BFloat16
    tf32=True,                                    # TF32 для ускорения
    logging_steps=5,
    max_steps=-1,                    
    save_strategy="epoch",
    optim="adamw_torch",                          # Стандартный AdamW (без 8bit)
    gradient_checkpointing=True,                  
    max_grad_norm=1.0,                            # Gradient clipping
    report_to="none",
    warmup_steps=10,                              # Разогрев оптимизатора
)

# 7. Инициализация SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print(f"\n{'='*60}")
print(f"Старт обучения на {torch.cuda.get_device_name(0)}")
print(f"VRAM доступно: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*60}\n")

# 8. Запуск обучения
trainer.train()

# 9. Сохранение финальных весов LORA
output_path = "/opt/train/qwen_final"
trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n{'='*60}")
print(f"✅ Обучение завершено!")
print(f"{'='*60}")
print(f"Адаптеры LORA сохранены в: {output_path}")
print(f"{'='*60}\n")
