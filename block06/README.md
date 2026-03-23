# Блок 06: Fine-tuning — адаптация модели под себя (LoRA + vLLM)

## Практическое задание

### Ресурсы

Вам предоставлена виртуальная машина с GPU-ускорителем (NVIDIA H100) и операционной системой Ubuntu 24.04.

- **Доступы:** SSH, права root (через `sudo`).
- **Предустановленное ПО:** Python 3.12, pip, драйверы NVIDIA (CUDA 12.x).

### Задача

Ваша задача — дообучить модель **Qwen/Qwen2.5-7B-Instruct** для общения в стиле пирата с помощью метода **LoRA**.

### Критерии успешного выполнения

- Виртуальное окружение Python создано в **`/opt/train/venv`**.
- Веса LoRA после обучения находятся в директории **`/opt/train/qwen_final`**.
- **vLLM** успешно запускается с дополнительными весами LoRA.

### LOG

В отчёт по заданию обычно включают фрагменты логов, подтверждающие выполнение критериев, например:

- успешное завершение `train.py` и наличие файлов в `/opt/train/qwen_final/` (`adapter_model.safetensors`, `adapter_config.json` и т.д.);
- вывод запуска `vllm serve` с `--enable-lora` и `--lora-modules ...=/opt/train/qwen_final`;
- при необходимости — короткий пример ответа API с включённым LoRA-модулем.

Курс на платформе REBRAIN: [Fine-Tuning: адаптация модели под себя](https://my.rebrainme.com/course/llm/task/3147).

---

## Описание решения в этой папке

Рабочий код и данные лежат в каталоге **`train/`**. Ниже — как это соотносится с заданием.

### Структура

| Файл / каталог | Назначение |
|----------------|------------|
| `train/train.py` | Дообучение **Qwen2.5-7B-Instruct** через **TRL `SFTTrainer`** и **PEFT `LoraConfig`** (BF16, без квантизации под H100). Чекпоинты: `/opt/train/qwen_pirate_v1/`, финальный адаптер и токенизатор сохраняются в **`/opt/train/qwen_final`**. |
| `train/dataset.py` | Генерация датасета в формате JSONL (диалоги в пиратском стиле). |
| `train/pirate_dataset.jsonl` | Готовый датасет для обучения. |
| `train/merge.py` | Опциональное слияние базовой модели с LoRA в полную весовую копию (для задания не обязательно). |
| `train/test_vllm.py` | Проверка инференса через vLLM с LoRA (`enable_lora`, `LoRARequest` на `/opt/train/qwen_final`). |
| `train/test_api.py` | Проверка OpenAI-совместимого API после запуска сервера. |
| `train/run_server.sh` | Пример запуска **`vllm serve`** с **`--enable-lora`** и **`--lora-modules pirate=/opt/train/qwen_final`**. |
| `train/requirements.txt` | Зависимости для venv (`torch`, `transformers`, `peft`, `trl`, `vllm` и др.). |

Каталог **`train1/`** — отдельный вариант/история эксперимента (в т.ч. под другой объём VRAM); для сдачи по описанию выше ориентируйтесь на **`train/`** и пути **`/opt/train/...`**.

### Типовой порядок действий на ВМ

1. Создать окружение: `sudo python3 -m venv /opt/train/venv`, активировать, установить зависимости из `train/requirements.txt` (рабочая директория — `train/`).
2. Запустить обучение: `python train.py` — по завершении появится **`/opt/train/qwen_final`**.
3. Запустить vLLM с LoRA, например командами из `train/run_server.sh` или `README.md` внутри `train/`, и при необходимости проверить `test_vllm.py` / `test_api.py`.

### Параметры LoRA и обучения (кратко)

В `train.py`: ранг LoRA **64**, **alpha 128**, целевые модули **`all-linear`**, несколько эпох SFT, batch и накопление градиента под H100, вывод в **`/opt/train/qwen_final`** через `trainer.save_model` и сохранение токенизатора.

Подробные шаги, команды сервера и разбор неполадок — в **`train/README.md`**, быстрый сценарий — в **`train/QUICKSTART.md`** и **`train/SUMMARY.md`**.
