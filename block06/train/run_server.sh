#!/bin/bash
# Скрипт запуска vLLM сервера с LORA адаптерами

echo "============================================================"
echo "🏴‍☠️ Запуск vLLM сервера с пиратской моделью"
echo "============================================================"

# Активация виртуального окружения
source /opt/train/venv/bin/activate

# Выберите один из вариантов:

# Вариант 1: С LORA адаптерами (рекомендуется)
echo "Запуск с LORA адаптерами..."
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --enable-lora \
    --lora-modules pirate=/opt/train/qwen_final \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# Вариант 2: Со слитой моделью (если выполнили merge.py)
# Раскомментируйте, если используете слитую модель:
# echo "Запуск со слитой моделью..."
# vllm serve /opt/train/qwen_merged_full \
#     --dtype bfloat16 \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.9 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 8000
