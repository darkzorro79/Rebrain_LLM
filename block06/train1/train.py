import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig

# Настройка прокси для загрузки моделей
# Раскомментируйте нужный вариант:

# Вариант 1: HTTP/HTTPS прокси
# os.environ['HTTP_PROXY'] = 'http://103.137.249.145:3128'
# os.environ['HTTPS_PROXY'] = 'http://103.137.249.145:3128'

# Вариант 2: SOCKS5 прокси (раскомментируйте, если нужен SOCKS5)
os.environ['HTTP_PROXY'] = 'socks5://103.137.249.145:3001'
os.environ['HTTPS_PROXY'] = 'socks5://103.137.249.145:3001'

# 1. Настройки модели
# Берем что-то серьезное. Qwen 2.5 7B или Llama 3.1 8B.
# На H200 можно взять даже Qwen2.5-32B-Instruct.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 2. Оптимизация под RTX 5090 (32GB VRAM)
# Для обучения 7B модели на 32GB нужна 4-bit квантизация
# Это позволит загрузить модель + оставить место для градиентов и активаций

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-битная квантизация (экономия ~75% памяти)
    bnb_4bit_quant_type="nf4",              # NF4 (Normal Float 4) - лучше для весов нейросетей
    bnb_4bit_compute_dtype=torch.bfloat16,  # Вычисления в bfloat16 (RTX 5090 оптимизирована под это)
    bnb_4bit_use_double_quant=True,         # Двойная квантизация - ещё больше экономии памяти
)

# 3. Загрузка модели с оптимизациями
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,             # BFloat16 для вычислений (было torch_dtype)
    device_map="auto"                 # Автоматическое распределение по GPU
)

# Включаем gradient checkpointing для экономии памяти
# Это замедлит обучение на ~20%, но сэкономит много памяти
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right" # Для SFTTrainer с packing=False нужен right

# Qwen и Llama 3 обычно имеют корректный pad_token, но на всякий случай:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Конфигурация LoRA (Оптимизирована под 32GB)
peft_config = LoraConfig(
    r=32,                # Уменьшили ранг для экономии памяти (32 - хороший баланс)
    lora_alpha=64,       # Alpha = r * 2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Учим все линейные слои для лучшего качества
    target_modules="all-linear"
)

# 5. Загружаем наш dataset
dataset = load_dataset("json", data_files="pirate_dataset.jsonl", split="train")

# Функция для применения шаблона модели (Qwen/Llama имеют свои теги)
def formatting_prompts_func(examples):
    # Извлекаем поле messages
    convos = examples["messages"]

    # --- БЛОК ПРОВЕРКИ ТИПА ---
    # Если convos - это список словарей (например, [{'role': 'user', ...}]),
    # значит нам пришел ОДИН пример, а не батч.
    if isinstance(convos, list) and len(convos) > 0 and isinstance(convos[0], dict):
        # Оборачиваем его в список, чтобы логика ниже сработала как для батча из 1 элемента
        convos = [convos]
    # --------------------------

    texts = []
    for messages in convos:
        # Теперь messages - это всегда список словарей (целый диалог)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    return texts

# 6. Параметры обучения (Оптимизированы под RTX 5090 32GB)
training_args = TrainingArguments(
    output_dir="./qwen_pirate_v1",
    num_train_epochs=3,              
    per_device_train_batch_size=4,   # Уменьшили до 4 для экономии памяти
    gradient_accumulation_steps=4,   # Увеличили до 4 (эффективный batch = 4*4 = 16)
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,                       # BFloat16 - оптимально для RTX 5090
    tf32=True,                       # TF32 для ускорения
    logging_steps=5,
    max_steps=-1,                    
    save_strategy="epoch",
    optim="adamw_8bit",              # 8-bit AdamW для экономии памяти оптимизатора
    gradient_checkpointing=True,     # Явно включаем gradient checkpointing
    max_grad_norm=1.0,               # Gradient clipping для стабильности
    report_to="none"                 
)

# 7. Инициализация SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print(f"Старт обучения на {torch.cuda.get_device_name(0)}")
trainer.train()

# Сохранение
trainer.save_model("./qwen_pirate_final")
print("Готово! Адаптеры сохранены.")
