# Блок 7: работа с промптами (инженерия запросов)

## Практическое задание

### Описание

Ваша задача — написать Python-скрипт, который с помощью LLM преобразует неструктурированные текстовые логи в формат JSON. Модель должна не просто переписать текст, а классифицировать важность события и извлечь коды ошибок.

### Ресурсы

- **API URL:** `https://llm.rebrainme.study`
- **API Key:** брать из переменной окружения `REBRAIN_API_KEY`
- **Модель:** `gemini-2.5-flash`
- **Входные данные:** файл с логами (можно сохранить в репозиторий с именем `logs.txt`). Пример содержимого:

```
[2023-10-21 14:02:01] AUTH_SERVICE: User login failed, incorrect password. Error code: 401.
DEBUG: Connection pool size is 10.
!!! FATAL_ERR !!! Database connection lost on db-prod-01. Immediate action required!
2023-10-21 14:05:00 - billing_app - WARNING: Latency is high (450ms).
[JUST_INFO] Backup started successfully.
nginx_proxy | 502 Bad Gateway | upstream timed out.
[2023-10-21 14:10:00] CRITICAL_FAIL: disk_usage_percent > 98% on /dev/sda1. Code: 999.
Minor issue: user avatar not loaded.
```

### Задача

Напишите скрипт `client.py`, который:

1. Скачивает файл (можно использовать библиотеку `requests`) и читает его содержимое.
2. Использует библиотеку `openai` для отправки запроса к LLM.
3. **System Prompt** задаёт роль «Log Parser».
4. **Few-Shot Prompting:** в сообщениях (`messages`) должен быть пример того, как превратить нестандартный лог (например, «SYSTEM DOWN») в уровень `CRITICAL`.
5. **Structured Output:** обязательно используйте `response_format={"type": "json_object"}`.
6. Результат выводит на экран (**stdout**).

Схема выходного JSON:

```json
{
  "entries": [
    {
      "service_name": "string",
      "error_code": "integer (0 если нет)",
      "severity": "CRITICAL | WARNING | INFO",
      "message": "string (очищенный текст ошибки)"
    }
  ]
}
```

### Критерии проверки

- Использование официальной библиотеки `openai`.
- Отсутствие захардкоженных API-ключей (использование `os.getenv`).
- Корректная обработка `severity` (только указанные 3 уровня).
- Наличие `response_format` в коде.

---

## Решение в этом репозитории

В каталоге уже есть рабочий пример `client.py`, который соответствует требованиям выше.

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Настройка

```bash
export REBRAIN_API_KEY="your_api_key_here"
```

### Запуск

```bash
python client.py
```

Скрипт скачивает логи по URL (в коде задан адрес сервера с `logs.txt`), отправляет их в LLM и печатает JSON на **stdout**. В репозитории также лежит локальный пример входного файла `logs.txt` для справки и самопроверки.

### Что реализовано в `client.py`

- библиотека `openai`, ключ из `REBRAIN_API_KEY`;
- `response_format={"type": "json_object"}`;
- system prompt с ролью Log Parser;
- few-shot с примером нестандартного лога → `CRITICAL`;
- уровни `severity`: только `CRITICAL`, `WARNING`, `INFO`.
