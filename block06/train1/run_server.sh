#!/bin/bash
# Запуск пиратской модели как OpenAI-совместимого API сервера через vLLM

echo "🏴‍☠️ Запуск Пиратской Модели как API сервера"
echo "=============================================="

MODEL_PATH="./qwen_pirate_merged_full"
PORT=8000

echo "Модель: $MODEL_PATH"
echo "Порт: $PORT"
echo ""
echo "После запуска сервер будет доступен по адресу:"
echo "  http://localhost:$PORT"
echo ""
echo "Документация API:"
echo "  http://localhost:$PORT/docs"
echo ""
echo "Совместимость с OpenAI API:"
echo "  Base URL: http://localhost:$PORT/v1"
echo ""
echo "=============================================="
echo ""

# Активируем виртуальное окружение
source .venv/bin/activate

# Запускаем vLLM сервер
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port $PORT \
    --trust-remote-code \
    --served-model-name "pirate-qwen-7b"

# Параметры:
# --model: путь к модели
# --dtype: тип данных (bfloat16 для RTX 5090)
# --max-model-len: максимальная длина контекста
# --gpu-memory-utilization: сколько VRAM использовать (0.9 = 90%)
# --port: порт для API
# --trust-remote-code: для Qwen моделей
# --served-model-name: имя модели в API
