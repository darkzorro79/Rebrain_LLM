# 🏴‍☠️ Обучение Пиратской Модели Qwen2.5-7B с LORA

Проект для дообучения модели Qwen/Qwen2.5-7B-Instruct в стиле пирата с использованием метода LORA на NVIDIA H100 96GB.

## 📋 Требования

- NVIDIA H100 96GB (или другой GPU с CUDA 12.x)
- Ubuntu 24.04
- Python 3.12
- CUDA 12.x драйверы

## 🚀 Быстрый старт

### 1. Создание виртуального окружения

```bash
# Создаем виртуальное окружение в указанной директории
sudo python3 -m venv /opt/train/venv

# Активируем окружение
source /opt/train/venv/bin/activate

# Обновляем pip
pip install --upgrade pip
```

### 2. Установка зависимостей

```bash
# Переходим в директорию проекта
cd /home/darkzorro/LLM_rebrain/block06/train

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 3. Подготовка данных

Датасет уже создан в файле `pirate_dataset.jsonl` (50 примеров диалогов в пиратском стиле).

Если нужно пересоздать или изменить датасет:

```bash
python dataset.py
```

### 4. Обучение модели

```bash
# Запускаем обучение
python train.py
```

**Важно:**
- Обучение займет примерно 10-20 минут на H100
- Чекпоинты сохраняются в `/opt/train/qwen_pirate_v1/`
- Финальные веса LORA сохраняются в `/opt/train/qwen_final/`

### 5. (Опционально) Слияние модели с адаптерами

Если нужна полная модель без адаптеров:

```bash
python merge.py
```

Слитая модель будет сохранена в `/opt/train/qwen_merged_full/`

### 6. Тестирование с vLLM

#### Вариант 1: Прямое использование LORA адаптеров (рекомендуется)

```bash
python test_vllm.py
```

Скрипт по умолчанию использует LORA адаптеры напрямую. Это экономит память и быстрее.

#### Вариант 2: Использование слитой модели

Отредактируйте `test_vllm.py`:

```python
USE_LORA = False
MODEL_PATH = "/opt/train/qwen_merged_full"
```

Затем запустите:

```bash
python test_vllm.py
```

### 7. Запуск API сервера (OpenAI-совместимый)

#### С LORA адаптерами:

```bash
source /opt/train/venv/bin/activate

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enable-lora \
    --lora-modules pirate=/opt/train/qwen_final \
    --host 0.0.0.0 \
    --port 8000
```

#### Со слитой моделью:

```bash
source /opt/train/venv/bin/activate

vllm serve /opt/train/qwen_merged_full \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
```

### 8. Тестирование API

В новом терминале:

```bash
source /opt/train/venv/bin/activate
python test_api.py
```

## 📁 Структура проекта

```
/home/darkzorro/LLM_rebrain/block06/train/
├── train.py              # Скрипт обучения с LORA
├── dataset.py            # Генератор датасета
├── pirate_dataset.jsonl  # Датасет (50 примеров)
├── merge.py              # Слияние базовой модели с LORA
├── test_vllm.py          # Тестирование через vLLM
├── test_api.py           # Тестирование API
├── requirements.txt      # Зависимости Python
└── README.md             # Этот файл

/opt/train/
├── venv/                 # Виртуальное окружение Python
├── qwen_pirate_v1/       # Чекпоинты во время обучения
├── qwen_final/           # ✅ Финальные веса LORA (требование задачи)
└── qwen_merged_full/     # (Опционально) Слитая модель
```

## 🎯 Критерии выполнения задачи

- ✅ venv создан в `/opt/train/venv`
- ✅ Веса LORA после обучения в `/opt/train/qwen_final`
- ✅ vLLM успешно запускается с LORA адаптерами

## 🔧 Технические детали

### Параметры обучения

- **Модель**: Qwen/Qwen2.5-7B-Instruct (7B параметров)
- **Метод**: LORA (Low-Rank Adaptation)
- **LORA rank**: 64 (увеличен для H100)
- **LORA alpha**: 128
- **Batch size**: 8 per device
- **Gradient accumulation**: 2 steps (эффективный batch = 16)
- **Learning rate**: 2e-4
- **Epochs**: 3
- **Precision**: BFloat16 (оптимально для H100)
- **Optimizer**: AdamW (стандартный, без квантизации)

### Оптимизация для H100 96GB

- **БЕЗ квантизации** - достаточно памяти для полной модели в BF16
- Увеличенный LORA rank (64 вместо 32)
- Больший batch size (8 вместо 4)
- Стандартный AdamW (без 8-bit)
- Gradient checkpointing для дополнительной оптимизации

### Использование памяти

- Базовая модель 7B в BF16: ~14GB
- LORA адаптеры: ~500MB
- Градиенты + оптимизатор: ~20-25GB
- **Итого**: ~40-45GB из 96GB доступно

## 🏴‍☠️ Примеры работы модели

```
Пользователь: Как дела?
Модель: Аррр, якорь мне в селезенку! Все отлично, паруса полны ветра!

Пользователь: Напиши функцию на Python
Модель: ```python
def hello(): pass
```
Держи, салага! Не забудь отступы, или пойдешь на корм акулам!

Пользователь: Что такое Docker?
Модель: Это как трюм твоего корабля, куда ты складываешь всё добро 
по контейнерам, чтобы крысы не погрызли зависимости!
```

## 🐛 Устранение неполадок

### Проблема: Out of Memory

Уменьшите `per_device_train_batch_size` в `train.py`:

```python
per_device_train_batch_size=4,  # Было 8
gradient_accumulation_steps=4,  # Было 2
```

### Проблема: Не запускается vLLM с LORA

Убедитесь, что используете актуальную версию vLLM:

```bash
pip install --upgrade vllm
```

### Проблема: Модель не скачивается

Настройте прокси в скриптах (если нужно):

```python
os.environ['HTTP_PROXY'] = 'socks5://ваш_прокси:порт'
os.environ['HTTPS_PROXY'] = 'socks5://ваш_прокси:порт'
```

## 📚 Дополнительная информация

### Расширение датасета

Отредактируйте `dataset.py`, добавив больше примеров в список `data`, затем:

```bash
python dataset.py
python train.py
```

### Изменение стиля модели

Измените примеры в `dataset.py` на другой стиль общения (формальный, молодежный, технический и т.д.).

### Использование другой модели

Измените `MODEL_NAME` в `train.py`:

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Пример
```

## 📞 Поддержка

При возникновении проблем проверьте:
1. Версию CUDA: `nvidia-smi`
2. Доступную память GPU: `nvidia-smi`
3. Логи обучения в `/opt/train/qwen_pirate_v1/`
4. Установленные пакеты: `pip list`

## 📄 Лицензия

Проект использует модель Qwen2.5-7B-Instruct (лицензия Apache 2.0).
