#!/usr/bin/env python3
"""
MCP Клиент для взаимодействия с LLM (Ollama) и MCP сервером.
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import Client


class MCPClient:
    """Клиент для работы с MCP сервером и Ollama LLM."""
    
    def __init__(self, server_script: str = "server.py"):
        self.server_script = server_script
        self.ollama_client = Client(host="http://localhost:11434")
        self.model = "qwen3.5:35b"
        self.session = None
        self.available_tools = []
        # Используем Python из виртуального окружения
        self.python_path = "/home/darkzorro/LLM_rebrain/block14/.venv/bin/python"
        
    async def connect_to_server(self):
        """Подключается к MCP серверу."""
        server_params = StdioServerParameters(
            command=self.python_path,
            args=[self.server_script],
        )
        
        self.stdio_context = stdio_client(server_params)
        self.read, self.write = await self.stdio_context.__aenter__()
        
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Получаем список доступных инструментов
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools
        
        print(f"✓ Подключено к MCP серверу")
        print(f"✓ Доступные инструменты: {[tool.name for tool in self.available_tools]}")
        
    async def disconnect(self):
        """Отключается от MCP сервера."""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(None, None, None)
            
    def _convert_tools_to_ollama_format(self):
        """Конвертирует инструменты MCP в формат Ollama."""
        ollama_tools = []
        for tool in self.available_tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            }
            ollama_tools.append(ollama_tool)
        return ollama_tools
    
    async def process_query(self, user_query: str) -> str:
        """Обрабатывает запрос пользователя через LLM и MCP инструменты."""
        
        # Формируем системный промпт
        system_prompt = """Ты - помощник для проверки состояния системных сервисов Linux.
У тебя есть доступ к инструменту check_service_status, который проверяет статус сервиса через systemctl.
Когда пользователь спрашивает о состоянии сервиса, используй этот инструмент.
Отвечай на русском языке кратко и информативно."""
        
        # Конвертируем инструменты в формат Ollama
        tools = self._convert_tools_to_ollama_format()
        
        # Отправляем запрос в Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        response = self.ollama_client.chat(
            model=self.model,
            messages=messages,
            tools=tools
        )
        
        # Обрабатываем ответ
        assistant_message = response.get("message", {})
        
        # Проверяем, хочет ли модель вызвать инструмент
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            # Выполняем вызовы инструментов
            tool_results = []
            
            for tool_call in assistant_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = tool_call["function"]["arguments"]
                
                print(f"→ Вызов инструмента: {function_name}({function_args})")
                
                # Вызываем инструмент на MCP сервере
                result = await self.session.call_tool(function_name, arguments=function_args)
                
                # Извлекаем текст из результата
                if result.content and len(result.content) > 0:
                    tool_result_text = result.content[0].text
                else:
                    tool_result_text = "Нет результата"
                    
                tool_results.append({
                    "role": "tool",
                    "content": tool_result_text
                })
                print(f"← Результат получен")
            
            # Добавляем сообщение ассистента с tool_calls
            messages.append(assistant_message)
            
            # Добавляем результаты инструментов
            for tool_call, tool_result in zip(assistant_message["tool_calls"], tool_results):
                messages.append({
                    "role": "tool",
                    "content": tool_result["content"]
                })
            
            # Получаем финальный ответ от LLM
            final_response = self.ollama_client.chat(
                model=self.model,
                messages=messages
            )
            
            return final_response.get("message", {}).get("content", "Не удалось получить ответ")
        
        # Если инструмент не нужен, возвращаем прямой ответ
        return assistant_message.get("content", "Не удалось получить ответ")
    
    async def run(self):
        """Запускает интерактивный цикл общения."""
        print("\n" + "="*50)
        print("MCP Клиент для проверки сервисов systemctl")
        print("Модель:", self.model)
        print("Для выхода введите: exit")
        print("="*50 + "\n")
        
        await self.connect_to_server()
        
        try:
            while True:
                try:
                    user_input = input("\n👤 Вы: ").strip()
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() == "exit":
                        print("\n👋 Завершение работы...")
                        break
                    
                    print("\n🤖 Обработка запроса...")
                    response = await self.process_query(user_input)
                    print(f"\n🤖 Ответ: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Завершение работы...")
                    break
                except Exception as e:
                    print(f"\n❌ Ошибка: {e}")
                    
        finally:
            await self.disconnect()


async def main():
    client = MCPClient("server.py")
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())