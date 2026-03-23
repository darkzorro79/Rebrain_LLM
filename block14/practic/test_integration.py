#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции MCP клиента с Ollama.
"""

import asyncio
import sys
sys.path.insert(0, "/home/darkzorro/LLM_rebrain/block14/practic")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import Client


async def test_full_integration():
    """Тестирует полную интеграцию MCP клиента с Ollama."""
    print("=" * 50)
    print("Тестирование интеграции MCP + Ollama")
    print("=" * 50)
    
    python_path = "/home/darkzorro/LLM_rebrain/block14/.venv/bin/python"
    ollama_client = Client(host="http://localhost:11434")
    model = "qwen3.5:35b"
    
    server_params = StdioServerParameters(
        command=python_path,
        args=["server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Получаем список инструментов
            tools_list = await session.list_tools()
            tools = tools_list.tools
            print(f"\n✓ Доступные инструменты: {[tool.name for tool in tools]}")
            
            # Конвертируем инструменты в формат Ollama
            ollama_tools = []
            for tool in tools:
                ollama_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema
                    }
                }
                ollama_tools.append(ollama_tool)
            
            # Тестовый запрос
            user_query = "Проверь статус сервиса docker"
            print(f"\n👤 Запрос: {user_query}")
            
            system_prompt = """Ты - помощник для проверки состояния системных сервисов Linux.
У тебя есть доступ к инструменту check_service_status, который проверяет статус сервиса через systemctl.
Когда пользователь спрашивает о состоянии сервиса, используй этот инструмент.
Отвечай на русском языке кратко и информативно."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            print("\n🤖 Отправка запроса в Ollama...")
            response = ollama_client.chat(
                model=model,
                messages=messages,
                tools=ollama_tools
            )
            
            assistant_message = response.get("message", {})
            print(f"🤖 Ответ модели: {assistant_message.get('content', '(нет контента)')}")
            
            # Проверяем, хочет ли модель вызвать инструмент
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                print(f"\n→ Модель хочет вызвать инструменты!")
                
                for tool_call in assistant_message["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    function_args = tool_call["function"]["arguments"]
                    
                    print(f"→ Вызов: {function_name}({function_args})")
                    
                    # Вызываем инструмент на MCP сервере
                    result = await session.call_tool(function_name, arguments=function_args)
                    
                    if result.content and len(result.content) > 0:
                        tool_result_text = result.content[0].text
                        print(f"← Результат получен ({len(tool_result_text)} символов)")
                        
                        # Формируем финальный ответ
                        messages.append(assistant_message)
                        messages.append({
                            "role": "tool",
                            "content": tool_result_text
                        })
                        
                        print("\n🤖 Получение финального ответа...")
                        final_response = ollama_client.chat(
                            model=model,
                            messages=messages
                        )
                        
                        final_content = final_response.get("message", {}).get("content", "")
                        print(f"\n🤖 Финальный ответ:\n{final_content}")
            else:
                print("\n✓ Модель не требует вызова инструментов")
    
    print("\n✓ Тест интеграции завершен успешно!")


if __name__ == "__main__":
    asyncio.run(test_full_integration())