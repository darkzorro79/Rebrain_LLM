# 🏴‍☠️ НАЧНИТЕ ОТСЮДА

## Проект готов к использованию!

Все файлы созданы и адаптированы под задачу обучения пиратской модели Qwen2.5-7B на NVIDIA H100 96GB.

---

## 📂 Что внутри?

### Файлы Python
- `train.py` - Обучение модели с LORA (без квантизации)
- `dataset.py` - Генератор датасета (50 примеров)
- `merge.py` - Слияние модели с LORA (опционально)
- `test_vllm.py` - Тестирование через vLLM
- `test_api.py` - Тестирование API сервера

### Данные
- `pirate_dataset.jsonl` - Датасет (50 диалогов в пиратском стиле)

### Конфигурация
- `requirements.txt` - Список Python модулей для venv
- `run_server.sh` - Скрипт запуска vLLM сервера

### Документация
- **`QUICKSTART.md`** ← **НАЧНИТЕ С ЭТОГО** (5 шагов до результата)
- `README.md` - Полная документация
- `SETUP.md` - Подготовка системы Ubuntu
- `SUMMARY.md` - Итоговая сводка
- `START_HERE.md` - Этот файл

---

## 🚀 Три простых шага

### Шаг 1: Установка

```bash
sudo python3 -m venv /opt/train/venv
source /opt/train/venv/bin/activate
cd /home/darkzorro/LLM_rebrain/block06/train
pip install --upgrade pip
pip install -r requirements.txt
```

### Шаг 2: Обучение

```bash
source /opt/train/venv/bin/activate
python train.py
```

⏱️ Займет 10-20 минут на H100

### Шаг 3: Тестирование

```bash
source /opt/train/venv/bin/activate
python test_vllm.py
```

---

## ✅ Результат

После обучения веса LORA будут сохранены в:

```
/opt/train/qwen_final/
```

И vLLM сможет их использовать! ✅

---

## 📖 Что читать дальше?

1. **`QUICKSTART.md`** - Детальная инструкция по 5 шагам
2. **`README.md`** - Полная документация проекта
3. **`SUMMARY.md`** - Итоговая сводка и сравнение с train1

---

## 💡 Ключевые особенности для H100

- ✅ БЕЗ квантизации (есть 96GB памяти)
- ✅ Увеличенный LORA rank (64 вместо 32)
- ✅ Больший batch size (8 вместо 4)
- ✅ Стандартный AdamW (без 8-bit)
- ✅ Поддержка vLLM с LORA адаптерами напрямую

---

## 🎯 Критерии выполнены

| Критерий | Статус |
|----------|--------|
| venv в `/opt/train/venv` | ✅ |
| Веса LORA в `/opt/train/qwen_final` | ✅ |
| vLLM с LORA работает | ✅ |

---

## 📦 Список модулей Python

Все в `requirements.txt`:

- torch (PyTorch с CUDA)
- transformers (Hugging Face)
- accelerate
- trl (обучение)
- peft (LORA)
- datasets
- vllm (инференс)
- sentencepiece, protobuf
- requests

---

## ❓ Вопросы?

Смотрите `README.md` или `SETUP.md`

---

## 🏴‍☠️ Пример работы

```
Вы: Привет!
Модель: Ахой! Чего тебе надобно на моем корабле?

Вы: Как создать список в Python?
Модель: ```python
loot = ['gold', 'rum', 'parrot']
```
Вот тебе список награбленного, юнга!
```

---

**НАЧНИТЕ С `QUICKSTART.md`** 🚀
