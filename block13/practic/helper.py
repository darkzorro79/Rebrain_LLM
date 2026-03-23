import os
import json
import subprocess
from openai import OpenAI

# Настраиваем клиента, API ключ берём из переменной окружения
client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://llm.rebrainme.study/"
)

def run_shell_command(command: str) -> str:
    """
    Выполняет shell-команду с подтверждением пользователя.
    Выводит команду и запрашивает подтверждение перед выполнением.
    """
    print(f"\nКоманда для выполнения: {command}")
    
    # Запрос подтверждения у пользователя
    while True:
        confirmation = input("Подтвердить выполнение? (y/n): ").strip().lower()
        if confirmation in ('y', 'yes', 'д', 'да'):
            break
        elif confirmation in ('n', 'no', 'н', 'нет'):
            return "Выполнение отменено пользователем."
        else:
            print("Введите 'y' для подтверждения или 'n' для отмены.")
    
    print(f"--- Выполняю: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "Команда выполнена успешно (нет вывода)."
        return f"Ошибка (код {result.returncode}): {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышено время ожидания выполнения команды."
    except Exception as e:
        return f"Ошибка выполнения: {str(e)}"


# Описание инструмента для Tool Calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Выполняет Linux shell команды. Используй для работы с файлами, директориями, процессами и системой.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell команда для выполнения, например 'ls -la' или 'cat /etc/hostname'"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

available_functions = {"run_shell_command": run_shell_command}

# Системный промпт
SYSTEM_PROMPT = """Ты — помощник системного администратора. Твоя задача — переводить текстовые запросы пользователя в консольные команды Linux.

Правила:
1. Анализируй запрос пользователя и определяй, какие команды нужны для его выполнения.
2. Используй функцию run_shell_command для выполнения команд.
3. После получения результата анализируй его и при необходимости выполняй дополнительные команды.
4. Когда задача выполнена, давай понятный итоговый ответ пользователю.
5. Будь осторожен с опасными командами (rm, chmod, chown и т.д.) — предупреждай пользователя о возможных рисках.

Всегда проверяй результат своих действий и сообщай пользователю о статусе выполнения."""


def process_request(user_request: str) -> None:
    """Обрабатывает один запрос пользователя."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    
    while True:
        # Запрос к модели
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Проверка: хочет ли модель вызвать инструменты?
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\n[Модель хочет выполнить: {function_name}]")
                
                # Выполняем функцию (с подтверждением пользователя)
                function_response = function_to_call(**function_args)
                
                # Добавляем результат в историю
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
            
            # Продолжаем цикл для обработки результатов
            continue
        
        else:
            # Модель дала финальный ответ
            print(f"\nОтвет: {response_message.content}")
            break


def main():
    """Главный цикл интерактивного помощника."""
    print("=" * 60)
    print("Помощник системного администратора")
    print("Введите ваш запрос на естественном языке.")
    print("Для выхода введите 'exit' или 'quit'")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nВаш запрос: ").strip()
            
            # Проверка на выход
            if user_input.lower() in ('exit', 'quit', 'выход', 'q'):
                print("До свидания!")
                break
            
            # Пропуск пустых запросов
            if not user_input:
                continue
            
            # Обработка запроса
            process_request(user_input)
            
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")


if __name__ == "__main__":
    main()