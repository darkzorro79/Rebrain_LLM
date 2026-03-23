import subprocess
from fastmcp import FastMCP

# Инициализация
mcp = FastMCP("SysAdmin Agent")

# Константы
FORBIDDEN_COMMANDS = ["rm", "mkfs", "dd", ":(){ :|:& };:"]

# Имитация системных логов (для демонстрации на любой ОС)
MOCK_LOGS = """
Jan 20 10:00:01 server systemd[1]: Started Session 1 of user root.
Jan 20 10:05:23 server sshd[1234]: Accepted publickey for user admin from 192.168.1.5
Jan 20 10:15:00 server kernel: [12345.67] Out of memory: Kill process 999 (python) score 850 or sacrifice child
Jan 20 10:15:01 server systemd[1]: main-service.service: Main process exited, code=killed, status=9/KILL
Jan 20 10:15:05 server systemd[1]: main-service.service: Failed with result 'signal'.
"""

# --- 1. RESOURCES (Ресурсы) ---
# Мы объявляем URI схему 'system://logs'.
# Модель может запросить чтение этого ресурса.

@mcp.resource("system://logs")
def get_system_logs() -> str:
    """
    Возвращает последние записи системного журнала.
    Читает данные из виртуального /var/log/syslog.
    """
    # В реальном продакшене здесь было бы:
    # with open("/var/log/syslog") as f: return f.read()
    return MOCK_LOGS.strip()


# --- 2. TOOLS (Инструменты) ---
# Исполнитель команд

@mcp.tool()
def execute_shell_command(command: str) -> str:
    """
    Выполняет безопасную shell-команду.
    Используй для проверки статуса сервисов или диагностики.
    """
    for bad_cmd in FORBIDDEN_COMMANDS:
        if bad_cmd in command:
            return f"Error: Command '{bad_cmd}' is forbidden."

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if not output and result.stderr:
            output = f"Error: {result.stderr.strip()}"
        elif not output:
            output = "Success (no output)"
        return output
    except Exception as e:
        return f"Execution failed: {str(e)}"


# --- 3. PROMPTS (Промпты) ---
# Шаблоны для пользователя

@mcp.prompt()
def analyze_incident() -> str:
    """
    Шаблон для анализа системных сбоев.
    Автоматически направляет внимание модели на логи.
    """
    # Мы возвращаем текст сообщения, которое будет отправлено модели от имени пользователя.
    # Обратите внимание: мы явно указываем URI ресурса.
    return (
        "Пожалуйста, проанализируй содержимое ресурса system://logs. "
        "1. Найди критические ошибки (OOM, Segfault, Failed services). "
        "2. Если ошибка найдена, предложи команду для проверки статуса упавшего сервиса "
        "с помощью инструмента execute_shell_command."
    )

if __name__ == "__main__":
    mcp.run(transport="http")
