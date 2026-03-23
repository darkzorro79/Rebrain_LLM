#!/bin/bash

# Проверка API /v1/models: GET к http://localhost:8080/v1/models (llama-server)
# Ответ должен содержать Qwen2.5-7B-Instruct-Q4_K_M.gguf
# При необходимости задайте API_KEY (Bearer), если сервер запущен с --api-key

ENDPOINT="http://localhost:8080/v1/models"
EXPECTED_MODEL="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
# API_KEY="my-secret-token"
API_KEY=""

# Проверка доступности endpoint с авторизацией
if ! curl -s --connect-timeout 5 -f -H "Authorization: Bearer $API_KEY" "$ENDPOINT" > /dev/null; then
    echo "[API /v1/models]: Ошибка: endpoint $ENDPOINT недоступен или не отвечает с авторизацией"
    exit 1
fi

# Получение и проверка ответа
echo "[API /v1/models]: GET-запрос к $ENDPOINT с авторизацией Bearer $API_KEY"
response=$(curl -s -H "Authorization: Bearer $API_KEY" "$ENDPOINT")

# Проверка, что ответ содержит модель в поле "id"
if echo "$response" | grep -q 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'; then
    echo "[API /v1/models]: Успешно: GET-запрос возвращает JSON со списком, содержащим $EXPECTED_MODEL"
    echo "Ответ: $response"
    exit 0
else
    echo "[API /v1/models]: Ошибка: ответ не содержит ожидаемую модель $EXPECTED_MODEL"
    echo "Ответ: $response"
    exit 1
fi
