import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Securely load environment variables from .env file
load_dotenv()

# Verify critical keys exist before starting
if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN"):
    logger.error("Missing critical environment variables. Check your .env file.")
    exit(1)

# Initialize Anthropics client
ai_client = Anthropic()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def connect_and_execute(user_prompt: str):
    """
    Connects to Grafana via MCP with retry logic.
    If the connection drops, it will automatically wait and try again up to 3 times.
    """
    logger.info("Initializing MCP Connection to Grafana...")

    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-grafana"],
        env={
            "GRAFANA_URL": os.getenv("GRAFANA_URL"),
            "GRAFANA_SERVICE_ACCOUNT_TOKEN": os.getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN"),
            "PATH": os.getenv("PATH", "")
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("MCP Connection Established Successfully.")
            
            mcp_tools = await session.list_tools()
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "No description provided.",
                    "input_schema": tool.inputSchema
                }
                for tool in mcp_tools.tools
            ]
            
            logger.info(f"Processing Request: '{user_prompt}'")
            
            # Send context to the LLM
            response = ai_client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=2048,
                tools=anthropic_tools,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Action Loop
            if response.stop_reason == "tool_use":
                tool_use = next(block for block in response.content if block.type == "tool_use")
                logger.info(f"Agent executing tool: {tool_use.name}")
                logger.debug(f"Payload: {json.dumps(tool_use.input)}") # Hidden unless debug mode is on
                
                # Execute the tool
                result = await session.call_tool(tool_use.name, arguments=tool_use.input)
                
                for res in result.content:
                    logger.info(f"Grafana API Response: {res.text}")
                    
            else:
                text_block = next((block for block in response.content if block.type == "text"), None)
                if text_block:
                    logger.info(f"Agent Response: {text_block.text}")

if __name__ == "__main__":
    prompt = "Create a new Grafana dashboard tracking network ping latency to 8.8.8.8"
    try:
        asyncio.run(connect_and_execute(prompt))
    except Exception as e:
        logger.critical(f"Agent failed after multiple retries. Error: {str(e)}")
