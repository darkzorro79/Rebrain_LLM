#!/usr/bin/env python3
"""
MCP Server для проверки состояния системных сервисов через systemctl.
"""

import subprocess
from fastmcp import FastMCP

# Создаем MCP сервер
mcp = FastMCP("systemctl-service-checker")


@mcp.tool()
def check_service_status(service_name: str) -> str:
    """
    Проверяет состояние системного сервиса через systemctl status.
    
    Args:
        service_name: Имя сервиса для проверки (например, 'nginx', 'docker', 'sshd')
    
    Returns:
        Строка с результатом выполнения systemctl status для указанного сервиса
    """
    try:
        result = subprocess.run(
            ["systemctl", "status", service_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Объединяем stdout и stderr для полного вывода
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
            
        if not output.strip():
            output = f"Сервис '{service_name}' не найден или нет информации."
            
        return output
        
    except subprocess.TimeoutExpired:
        return f"Ошибка: Превышено время ожидания при проверке сервиса '{service_name}'"
    except FileNotFoundError:
        return "Ошибка: Команда systemctl не найдена в системе"
    except Exception as e:
        return f"Ошибка при проверке сервиса '{service_name}': {str(e)}"


if __name__ == "__main__":
    # Запускаем сервер
    mcp.run()