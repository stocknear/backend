import asyncio
import os
import orjson
from agents import Agent, Runner
from agents.mcp import MCPServer, MCPServerSse
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from utils.helper import load_latest_json, json_to_string, process_request

#load .env
load_dotenv()

with open("json/llm/instructions.json","rb") as file:
    INSTRUCTIONS = json_to_string(orjson.loads(file.read()))


async def main():
    
    server = MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8000/sse",
        },
    )
    await server.connect()  # Initialize the server connection
    
    agent = Agent(
        name="Assistant",
        model="gpt-4.1-nano-2025-04-14",
        instructions=INSTRUCTIONS,
        mcp_servers=[server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    
    message = "list stocks with the highest market cap in the energy sector"
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    await server.cleanup()  # Clean up the server connection

if __name__ == "__main__":

    asyncio.run(main())