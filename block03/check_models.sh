#!/bin/bash

# Проверка API /v1/models: GET-запрос к http://localhost:8000/v1/models
# Должен возвращать JSON со списком, содержащим Qwen/Qwen2.5-0.5B-Instruct
# Используется авторизация с --api-key "my-secret-token"

ENDPOINT="http://localhost:8000/v1/models"
EXPECTED_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
API_KEY="my-secret-token"

# Проверка доступности endpoint с авторизацией
if ! curl -s --connect-timeout 5 -f -H "Authorization: Bearer $API_KEY" "$ENDPOINT" > /dev/null; then
    echo "[API /v1/models]: Ошибка: endpoint $ENDPOINT недоступен или не отвечает с авторизацией"
    exit 1
fi

# Получение и проверка ответа
echo "[API /v1/models]: GET-запрос к $ENDPOINT с авторизацией Bearer $API_KEY"
response=$(curl -s -H "Authorization: Bearer $API_KEY" "$ENDPOINT")

# Проверка, что ответ содержит модель в поле "id"
if echo "$response" | grep -q '"id":"Qwen/Qwen2.5-0.5B-Instruct"'; then
    echo "[API /v1/models]: Успешно: GET-запрос возвращает JSON со списком, содержащим $EXPECTED_MODEL"
    echo "Ответ: $response"
    exit 0
else
    echo "[API /v1/models]: Ошибка: ответ не содержит ожидаемую модель $EXPECTED_MODEL"
    echo "Ответ: $response"
    exit 1
fi
