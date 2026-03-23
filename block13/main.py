import os
import json
import subprocess
from openai import OpenAI

# Настраиваем клиента
client = OpenAI(
    api_key="API_KEY",
    base_url="https://llm.rebrainme.study/"
)

def execute_shell_command(command: str) -> str:
    """Выполняет shell-команду и возвращает её вывод."""
    print(f"--- Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error (Code {result.returncode}): {result.stderr.strip()}"
    except Exception as e:
        return f"System Execution Error: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Выполняет команды Linux Shell. Используй для проверки файлов и системы.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Команда, например 'ls -la'"}
                },
                "required": ["command"]
            }
        }
    }
]

available_functions = {"execute_shell_command": execute_shell_command}

messages = [
    {
        "role": "system",
        "content": "Ты — системный администратор. Если тебе нужно получить информацию (список файлов, содержимое), вызывай инструменты. Всегда проверяй результат своих действий. После получения результата проанализируй его и реши, нужно ли ещё что-то сделать или можно давать финальный ответ."
    },
    {
        "role": "user",
        "content": "Найди самый большой файл в /var/log директории файл с расширением .log, и выведи его последние 5 строк"
    }
]

print("Запуск цикла обработки...")

while True:
    # 1. Запрос к модели
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Добавляем ответ модели в историю сообщений
    # Важно: если в ответе есть tool_calls, объект сообщения должен быть в истории перед результатами функций
    messages.append(response_message)

    # 2. Проверка: хочет ли модель вызвать инструменты?
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            print(f"Модель вызывает: {function_name}({function_args})")

            # Выполняем действие
            function_response = function_to_call(**function_args)

            # Добавляем результат выполнения в историю
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # После того как все функции в текущем шаге выполнены, 
        # цикл while уходит на следующую итерацию, чтобы отправить результаты модели
        continue

    else:
        # Если tool_calls нет, значит модель просто ответила пользователю
        print("\n--- ФИНАЛЬНЫЙ ОТВЕТ ---")
        print(response_message.content)
        break # Выход из цикла
