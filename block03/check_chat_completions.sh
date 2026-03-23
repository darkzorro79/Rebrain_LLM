#!/bin/bash
set -euo pipefail

ENDPOINT="http://localhost:8000/v1/chat/completions"
API_KEY="my-secret-token"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROMPT="Hello"

PAYLOAD="{
  \"model\": \"$MODEL\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"$PROMPT\"}
  ],
  \"temperature\": 0.7
}"

echo "[API /v1/chat/completions]: POST-запрос к $ENDPOINT с авторизацией Bearer $API_KEY"
echo "Тело запроса: $PAYLOAD"

# Сохраним тело отдельно, статус отдельно
body_file="$(mktemp)"
http_code=$(
  curl -sS -o "$body_file" -w "%{http_code}" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -X POST \
    -d "$PAYLOAD" \
    "$ENDPOINT"
)

if [ "$http_code" -ne 200 ]; then
  echo "[API /v1/chat/completions]: Ошибка: ожидается код 200, получен $http_code"
  echo "Ответ:"
  cat "$body_file"
  exit 1
fi

# Простая проверка наличия content (без jq)
if grep -q '"content"[[:space:]]*:[[:space:]]*"' "$body_file"; then
  echo "[API /v1/chat/completions]: Успешно: POST-запрос возвращает JSON с кодом 200 OK и сгенерированным текстом"
  echo "Ответ:"
  cat "$body_file"
  exit 0
else
  echo "[API /v1/chat/completions]: Ошибка: ответ не содержит сгенерированный текст"
  echo "Ответ:"
  cat "$body_file"
  exit 1
fi
