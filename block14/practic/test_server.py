#!/usr/bin/env python3
"""
Тестовый скрипт для проверки MCP сервера и клиента.
"""

import asyncio
import sys
sys.path.insert(0, "/home/darkzorro/LLM_rebrain/block14/practic")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Тестирует MCP сервер напрямую."""
    print("=" * 50)
    print("Тестирование MCP сервера")
    print("=" * 50)
    
    python_path = "/home/darkzorro/LLM_rebrain/block14/.venv/bin/python"
    
    server_params = StdioServerParameters(
        command=python_path,
        args=["server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Получаем список инструментов
            tools_list = await session.list_tools()
            print(f"\n✓ Доступные инструменты: {[tool.name for tool in tools_list.tools]}")
            
            # Тестируем инструмент check_service_status
            print("\n→ Тестирование check_service_status('docker')...")
            result = await session.call_tool("check_service_status", arguments={"service_name": "docker"})
            
            if result.content:
                print("← Результат:")
                print(result.content[0].text[:500])
            
            print("\n→ Тестирование check_service_status('nginx')...")
            result = await session.call_tool("check_service_status", arguments={"service_name": "nginx"})
            
            if result.content:
                print("← Результат:")
                print(result.content[0].text[:500])
    
    print("\n✓ Тест MCP сервера завершен успешно!")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())