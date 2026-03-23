# 🚀 Быстрый старт - 5 шагов

## Шаг 1: Создание venv и установка зависимостей

```bash
# Создаем venv в нужной директории
sudo python3 -m venv /opt/train/venv

# Активируем
source /opt/train/venv/bin/activate

# Обновляем pip
pip install --upgrade pip

# Переходим в директорию проекта
cd /home/darkzorro/LLM_rebrain/block06/train

# Устанавливаем все зависимости
pip install -r requirements.txt
```

## Шаг 2: Обучение модели

```bash
# Активируем окружение (если еще не активировано)
source /opt/train/venv/bin/activate

# Запускаем обучение (займет ~10-20 минут на H100)
python train.py
```

**Результат:**
- Веса LORA сохранятся в `/opt/train/qwen_final/` ✅

## Шаг 3: Тестирование с vLLM (напрямую в Python)

```bash
# Активируем окружение
source /opt/train/venv/bin/activate

# Запускаем тест (поддерживает интерактивный режим)
python test_vllm.py
```

## Шаг 4: (Опционально) Запуск API сервера

```bash
# В отдельном терминале
source /opt/train/venv/bin/activate
./run_server.sh
```

## Шаг 5: Тестирование API

```bash
# В другом терминале
source /opt/train/venv/bin/activate
python test_api.py
```

---

## 📋 Список зависимостей Python

Все зависимости указаны в `requirements.txt`:

- **torch** - PyTorch для обучения
- **transformers** - Hugging Face библиотека моделей
- **accelerate** - Ускорение обучения
- **trl** - Transformer Reinforcement Learning
- **peft** - Parameter-Efficient Fine-Tuning (LORA)
- **datasets** - Работа с датасетами
- **vllm** - Быстрый инференс
- **sentencepiece, protobuf** - Токенизация
- **requests** - HTTP запросы для API

---

## ✅ Проверка критериев

После выполнения всех шагов:

1. ✅ **venv создан**: `/opt/train/venv`
2. ✅ **Веса LORA**: `/opt/train/qwen_final/`
3. ✅ **vLLM работает**: `python test_vllm.py` успешно запускается с LORA

---

## 🎯 Дополнительные команды

### Проверка GPU
```bash
nvidia-smi
```

### Проверка установленных пакетов
```bash
source /opt/train/venv/bin/activate
pip list
```

### Слияние модели с LORA (если нужна полная модель)
```bash
source /opt/train/venv/bin/activate
python merge.py
```

Результат будет в `/opt/train/qwen_merged_full/`

---

## 🏴‍☠️ Пример диалога

```
Вы: Привет!
🏴‍☠️ Модель: Ахой! Чего тебе надобно на моем корабле?

Вы: Как создать список в Python?
🏴‍☠️ Модель: ```python
loot = ['gold', 'rum', 'parrot']
```
Вот тебе список награбленного, юнга!
```

---

## ❓ Проблемы?

Смотрите полную документацию в `README.md`
