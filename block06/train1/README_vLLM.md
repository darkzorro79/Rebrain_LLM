# 🏴‍☠️ Запуск Пиратской Модели через vLLM

## Что у нас есть

После успешного обучения и слияния у вас есть:
- ✅ Обученная модель: `./qwen_pirate_merged_full/`
- ✅ Адаптеры LoRA: `./qwen_pirate_final/`
- ✅ vLLM установлен и готов к использованию

## Варианты запуска

### Вариант 1: Python скрипт (рекомендуется для тестирования)

Быстрое тестирование модели в интерактивном режиме:

```bash
python3 test_vllm.py
```

**Что делает:**
- Загружает модель через vLLM (быстрая генерация)
- Запускает автоматические тесты
- Переходит в интерактивный режим для общения

**Плюсы:**
- Просто и быстро
- Интерактивный режим
- Не требует запуска сервера

### Вариант 2: API сервер (рекомендуется для продакшена)

Запуск модели как OpenAI-совместимого API сервера:

```bash
./run_server.sh
```

**Что делает:**
- Запускает vLLM сервер на порту 8000
- Предоставляет REST API совместимый с OpenAI
- Поддерживает multiple requests, batching, streaming

**Доступные endpoints:**
- `http://localhost:8000/v1/chat/completions` - Chat API
- `http://localhost:8000/v1/completions` - Completion API
- `http://localhost:8000/docs` - Swagger документация
- `http://localhost:8000/health` - Health check

**Тестирование API сервера:**

В отдельном терминале запустите:

```bash
python3 test_api.py
```

### Вариант 3: Использование в своем коде

#### Python с vLLM:

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Загрузка модели
llm = LLM(
    model="./qwen_pirate_merged_full",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("./qwen_pirate_merged_full")

# Генерация
messages = [{"role": "user", "content": "Привет!"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)
```

#### Запросы к API через curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pirate-qwen-7b",
    "messages": [{"role": "user", "content": "Привет!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

#### Запросы через OpenAI Python SDK:

```python
from openai import OpenAI

# Указываем наш локальный сервер
client = OpenAI(
    api_key="EMPTY",  # vLLM не требует API ключ
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="pirate-qwen-7b",
    messages=[
        {"role": "user", "content": "Привет!"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

## Параметры производительности

В `run_server.sh` и `test_vllm.py` можно настроить:

- `--max-model-len`: Максимальная длина контекста (2048, 4096, 8192)
- `--gpu-memory-utilization`: Процент VRAM (0.8 = 80%, 0.9 = 90%)
- `--dtype`: Тип данных (bfloat16 оптимально для RTX 5090)
- `--tensor-parallel-size`: Количество GPU (если больше одной)

## Производительность

На RTX 5090 32GB ожидаемая производительность:

- **Throughput**: ~80-120 tokens/sec (зависит от batch size)
- **Latency**: ~20-50ms для первого токена
- **Memory**: ~12-16GB VRAM (модель + KV cache)

## Troubleshooting

### Ошибка: CUDA out of memory

Уменьшите `--gpu-memory-utilization` или `--max-model-len`:

```bash
--gpu-memory-utilization 0.7
--max-model-len 2048
```

### Медленная генерация

- Убедитесь, что используется bfloat16 (не float32)
- Проверьте загрузку GPU через `nvidia-smi`
- Увеличьте batch size в production

### Сервер не запускается

Проверьте, свободен ли порт 8000:

```bash
lsof -i :8000
```

Можно изменить порт в `run_server.sh`.

## Что дальше?

- 🚀 Деплой в продакшн (Docker, Kubernetes)
- 📊 Добавить мониторинг (Prometheus, Grafana)
- 🔧 Fine-tuning на специфичных данных
- 📈 Benchmarking и оптимизация
- ☁️ Загрузка на HuggingFace Hub

---

**Создано:** январь 2026  
**Модель:** Qwen2.5-7B-Instruct + LoRA (Pirate Fine-tune)  
**Hardware:** NVIDIA RTX 5090 32GB
