import asyncio
import json
from openai import OpenAI
from fastmcp import Client

# Настраиваем клиента OpenAI
client = OpenAI(
    api_key="API_KEY",
    base_url="https://llm.rebrainme.study/"
)

MCP_SERVER_URL = "http://localhost:8000/mcp"

async def get_mcp_tools(mcp: Client) -> list[dict]:
    """Получает список инструментов с MCP сервера и конвертирует в формат OpenAI."""
    tools_result = await mcp.list_tools()
    openai_tools = []
    for tool in tools_result:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
        })
    return openai_tools


async def call_mcp_tool(mcp: Client, name: str, arguments: dict) -> str:
    """Вызывает инструмент на MCP сервере и возвращает результат."""
    result = await mcp.call_tool(name, arguments)
    parts = [content.text for content in result.content if hasattr(content, "text")]
    return "\n".join(parts) if parts else "No output"

async def main():
    async with Client(MCP_SERVER_URL) as mcp:
        # Получаем инструменты с сервера
        tools = await get_mcp_tools(mcp)
        print(f"Загружено инструментов с MCP: {[t['function']['name'] for t in tools]}")

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — системный администратор. У тебя есть инструмент execute_shell_command "
                    "для выполнения команд. НИКОГДА не проси пользователя выполнить команды самостоятельно — "
                    "всегда выполняй их сам через инструмент. "
                    "Действуй автономно: сначала исследуй систему командами, затем давай финальный ответ."
                )
            },
            {
                "role": "user",
                "content": (
                    "Найди самый большой файл в /var/log директории с расширением .log "
                    "и выведи его последние 5 строк"
                )
            }
        ]

        print("Запуск цикла обработки...")

        while True:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"Модель вызывает: {function_name}({function_args})")

                    function_response = await call_mcp_tool(mcp, function_name, function_args)
                    print(f"--- Результат: {function_response[:200]}")

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })

                continue

            else:
                print("\n--- ФИНАЛЬНЫЙ ОТВЕТ ---")
                print(response_message.content)
                break


if __name__ == "__main__":
    asyncio.run(main())
