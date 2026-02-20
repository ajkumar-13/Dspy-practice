"""
Blog 6.3 - MCP Integration: Remote, Local, and Hybrid MCP Agents
Run: python 01_mcp_integration.py

Requires: uv add "dspy[mcp]" python-dotenv
"""

import dspy
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ============================================================
# Part 1: Remote MCP Server (HTTP)
# ============================================================

async def remote_mcp_demo():
    """Connect to a remote MCP server via Streamable HTTP."""
    server_url = "http://localhost:8080/mcp"

    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover available tools
            mcp_tools = await session.list_tools()
            print(f"Found {len(mcp_tools.tools)} MCP tools:")
            for tool in mcp_tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # Build a ReAct agent with MCP tools
            agent = dspy.ReAct(
                "question -> answer",
                tools=dspy_tools,
                max_iters=8,
            )

            result = await agent.acall(
                question="Search for recent news about AI agents"
            )
            print(f"\nAnswer: {result.answer}")


# ============================================================
# Part 2: Local MCP Server (Stdio)
# ============================================================

async def local_mcp_demo():
    """Connect to a local MCP server via stdio."""
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "./documents"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            print(f"Found {len(mcp_tools.tools)} tools from filesystem server")

            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            agent = dspy.ReAct(
                "question -> answer",
                tools=dspy_tools,
                max_iters=6,
            )

            result = await agent.acall(
                question="What files are in the documents directory?"
            )
            print(f"\nAnswer: {result.answer}")


# ============================================================
# Part 3: Hybrid Agent (MCP + Local Tools)
# ============================================================

async def build_hybrid_agent(session, mcp_tools_list):
    """Build an agent with both MCP and local tools."""

    # Convert MCP tools
    mcp_dspy_tools = [
        dspy.Tool.from_mcp_tool(session, tool)
        for tool in mcp_tools_list
    ]

    # Local tools
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    # Combine both
    all_tools = mcp_dspy_tools + [calculate]

    return dspy.ReAct(
        "question -> answer",
        tools=all_tools,
        max_iters=10,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MCP Integration Demo")
    print("=" * 60)
    print(
        "\nNote: This script requires a running MCP server."
        "\nTo test the remote demo, start an MCP server at http://localhost:8080/mcp"
        "\nTo test the local demo, ensure 'npx' is available and a ./documents dir exists."
        "\n"
    )

    # Uncomment the demo you want to run:
    # asyncio.run(remote_mcp_demo())
    # asyncio.run(local_mcp_demo())

    print("Uncomment the desired demo function in __main__ to run it.")
