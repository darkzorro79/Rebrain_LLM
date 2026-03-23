#!/bin/bash
# Запуск высокопроизводительного сервера инференса с vLLM для модели Qwen2.5-0.5B-Instruct
# Поддерживает OpenAI-совместимый API, работает в фоне, слушает 0.0.0.0:8000

# Установка vLLM
pip install vllm

# Запуск сервера vLLM в фоновом режиме с nohup
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "my-secret-token" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --dtype auto \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --disable-log-stats \
  > vllm_server.log 2>&1 &

# Вывод PID процесса
echo "vLLM сервер запущен в фоне. PID: $!"
echo "Логи доступны в файле vllm_server.log"
