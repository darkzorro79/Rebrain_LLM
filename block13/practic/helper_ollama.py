import os
import re
import json
import subprocess
from openai import OpenAI

# Настраиваем клиента для Ollama
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

MAX_ITERATIONS = 5  # Максимальное количество итераций в цикле


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


def is_valid_shell_command(text: str) -> bool:
    """Проверяет, похож ли текст на валидную shell команду."""
    if not text:
        return False
    
    # Пропускаем многострочный текст (это вывод команд, а не команда)
    if '\n' in text:
        return False
    
    # Пропускаем текст, который начинается с типичного вывода ls/файлов
    if text.startswith(('total ', 'drwx', '-rw', 'lrwx', '-rwx', 'crw', 'brw')):
        return False
    
    # Пропускаем текст похожий на вывод ls -la (строка с датой и временем)
    # Пример: "drwxrwxr-x 3 darkzorro darkzorro  4096 фев 22 02:13 ."
    if re.match(r'^[d\-l][rwx\-]{9}\s+\d+', text):
        return False
    
    # Список распространённых shell команд
    common_commands = [
        'ls', 'cat', 'cd', 'pwd', 'echo', 'grep', 'find', 'mkdir', 'rm', 'cp',
        'mv', 'touch', 'chmod', 'chown', 'ps', 'top', 'kill', 'df', 'du', 'free',
        'uname', 'hostname', 'whoami', 'id', 'date', 'cal', 'uptime', 'dmesg',
        'tail', 'head', 'less', 'more', 'wc', 'sort', 'uniq', 'diff', 'tar',
        'gzip', 'gunzip', 'zip', 'unzip', 'wget', 'curl', 'ping', 'netstat',
        'ss', 'ip', 'ifconfig', 'systemctl', 'service', 'journalctl', 'apt',
        'yum', 'dnf', 'pacman', 'snap', 'flatpak', 'docker', 'kubectl', 'git',
        'python', 'python3', 'pip', 'node', 'npm', 'make', 'gcc', 'g++', 'sudo',
        'man', 'which', 'whereis', 'type', 'file', 'stat', 'tree', 'nc', 'telnet',
        'scp', 'rsync', 'ssh', 'awk', 'sed', 'cut', 'tr', 'xargs', 'tee', 'env',
        'export', 'source', 'alias', 'history', 'clear', 'exit', 'sleep', 'time'
    ]
    
    # Проверяем, начинается ли текст с известной команды
    first_word = text.split()[0] if text.split() else ''
    return first_word in common_commands


def extract_command_from_text(text: str) -> str | None:
    """
    Извлекает команду из текста ответа модели.
    Ищет команды в обратных кавычках.
    """
    if not text:
        return None
    
    # Ищем команды в одинарных обратных кавычках
    backtick_matches = re.findall(r'`([^`\n]+)`', text)
    for match in backtick_matches:
        cmd = match.strip()
        if is_valid_shell_command(cmd):
            return cmd
    
    return None


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
SYSTEM_PROMPT = """Ты — помощник системного администратора. Отвечай кратко и по делу.

ПРАВИЛА:
1. Если пользователь просит выполнить команду — вызывай функцию run_shell_command.
2. Если пользователь просто общается (приветствие, вопрос) — отвечай текстом БЕЗ вызова функций.
3. После выполнения команды анализируй результат и давай краткий ответ.
4. НЕ вызывай функции без необходимости.
5. НЕ предлагай выполнить команды, если пользователь не просит.

ВАЖНО: На приветствия и общие вопросы отвечай просто текстом, не вызывая функции."""


def process_request(user_request: str) -> None:
    """Обрабатывает один запрос пользователя."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    
    iteration = 0
    command_was_executed = False  # Флаг: была ли уже выполнена команда
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        try:
            # Запрос к модели
            response = client.chat.completions.create(
                model="gpt-oss:20b",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Проверка: хочет ли модель вызвать инструменты?
            if response_message.tool_calls:
                any_command_executed = False
                
                for tool_call in response_message.tool_calls:
                    try:
                        # Получаем имя функции, очищая от возможных артефактов
                        function_name = tool_call.function.name
                        if '<|' in function_name:
                            function_name = function_name.split('<|')[0]
                        
                        if function_name not in available_functions:
                            print(f"\n[Неизвестная функция: {function_name}]")
                            continue
                        
                        # Парсим аргументы с обработкой ошибок
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            print(f"\n[Ошибка парсинга аргументов]")
                            continue
                        
                        # Проверяем, что команда есть и валидна
                        if 'command' not in function_args or not function_args['command']:
                            continue
                        
                        command = function_args['command']
                        if not is_valid_shell_command(command):
                            print(f"\n[Пропущена невалидная команда: {command[:80]}{'...' if len(command) > 80 else ''}]")
                            # Добавляем сообщение об ошибке в историю
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Ошибка: команда '{command}' не распознана как валидная shell команда. Попробуй другую команду.",
                            })
                            continue
                        
                        print(f"\n[Модель хочет выполнить: {command}]")
                        
                        # Выполняем функцию (с подтверждением пользователя)
                        function_response = available_functions[function_name](**function_args)
                        command_was_executed = True
                        any_command_executed = True
                        
                        # Показываем результат пользователю
                        print(f"\n--- Результат команды ---\n{function_response}\n------------------------")
                        
                        # Добавляем результат в историю
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        })
                    
                    except Exception as e:
                        print(f"\n[Ошибка при обработке: {e}]")
                        continue
                
                # Если ни одна команда не была выполнена, но были tool_calls - идём на следующую итерацию
                # чтобы модель могла предложить другие команды
                if not any_command_executed and response_message.tool_calls:
                    continue
                
                # После выполнения команды идём на следующую итерацию за ответом модели
                if any_command_executed:
                    continue
            
            # Если tool_calls нет — модель дала текстовый ответ
            elif response_message.content:
                content = response_message.content.strip()
                
                # Проверяем на команду в тексте ТОЛЬКО на первой итерации и если ещё ничего не выполняли
                if iteration == 1 and not command_was_executed:
                    extracted = extract_command_from_text(content)
                    request_lower = user_request.lower()
                    needs_execution = any(kw in request_lower for kw in 
                        ['выполни', 'запусти', 'покажи', 'выведи', 'найди', 'список', 
                         'run', 'execute', 'проверь', 'посмотри'])
                    
                    if extracted and needs_execution:
                        print(f"\n[Обнаружена команда в тексте: {extracted}]")
                        
                        confirmation = input("Выполнить эту команду? (y/n): ").strip().lower()
                        if confirmation in ('y', 'yes', 'д', 'да'):
                            result = run_shell_command(extracted)
                            command_was_executed = True
                            # Добавляем результат и просим модель прокомментировать
                            messages.append({
                                "role": "user",
                                "content": f"Результат выполнения команды:\n{result}\n\nПрокомментируй результат кратко."
                            })
                            continue
                
                # Выводим ответ модели
                print(f"\nОтвет: {content}")
                return
            
            else:
                print("\nОтвет: Модель не вернула ответ.")
                return
        
        except Exception as e:
            print(f"\n[Ошибка: {e}]")
            return
    
    # Достигнут лимит итераций
    print(f"\n[Достигнут лимит итераций ({MAX_ITERATIONS}). Завершение.]")


def main():
    """Главный цикл интерактивного помощника."""
    print("=" * 60)
    print("Помощник системного администратора (Ollama)")
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