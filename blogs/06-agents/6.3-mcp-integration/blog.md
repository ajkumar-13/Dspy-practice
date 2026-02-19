# 6.3 MCP Integration


## Introduction

So far, every tool we've built has been a Python function in our codebase. That works, but it means every team reinvents the wheel, writing their own search tools, file readers, database connectors, and API wrappers. The **Model Context Protocol (MCP)** solves this by providing an open, standardized protocol for connecting LM applications to external tools and data sources.

Think of MCP as a **USB port for AI tools**. Any MCP-compatible server exposes tools in a standard format, and any MCP-compatible client (including DSPy) can discover and use them. A growing ecosystem of pre-built MCP servers already exists, for GitHub, Slack, databases, file systems, web search, and more.

DSPy integrates with MCP out of the box. You can convert any MCP tool into a `dspy.Tool` and use it with `dspy.ReAct` or manual tool handling, just like a local function.

---

## What You'll Learn

- What MCP is and why it matters for agent ecosystems
- Installing MCP support in DSPy
- Connecting to MCP servers via HTTP (remote) and stdio (local)
- Converting MCP tools to DSPy tools with `dspy.Tool.from_mcp_tool()`
- Building a ReAct agent powered by MCP tools

---

## Prerequisites

- Completed [6.2 Advanced Tool Use](../6.2-advanced-tool-use/blog.md)
- DSPy with MCP extras installed (`uv add "dspy[mcp]" python-dotenv`)
- An MCP server to connect to (we'll cover setup options below)

---

## What Is MCP?

The **Model Context Protocol** is an open protocol (originally developed by Anthropic) that standardizes how AI applications communicate with external tools and data sources. An MCP server exposes:

- **Tools**: callable functions with typed parameters and descriptions
- **Resources**: structured data the model can read
- **Prompts**: reusable prompt templates

The key benefit: **write a tool once as an MCP server, use it from any MCP client** (DSPy, Claude Desktop, Cursor, or any other compatible application). This avoids the fragmentation of every framework having its own tool format.

---

## Installing MCP Support

DSPy's MCP integration requires the `mcp` extra:

```bash
uv add "dspy[mcp]"
```

This installs the `mcp` Python package alongside DSPy, providing the client libraries needed to connect to MCP servers.

---

## Connecting to Remote MCP Servers (HTTP)

Remote MCP servers communicate over HTTP using the Streamable HTTP transport. This is the most common pattern for production deployments. The server runs on a remote machine and your agent connects to it over the network.

```python

import asyncio
import dspy
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


async def main():
    # Connect to a remote MCP server via HTTP
    server_url = "http://localhost:8080/mcp"

    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover available tools on the server
            mcp_tools = await session.list_tools()
            print(f"Found {len(mcp_tools.tools)} MCP tools:")
            for tool in mcp_tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Convert MCP tools to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # Build a ReAct agent with the MCP tools
            agent = dspy.ReAct(
                "question -> answer",
                tools=dspy_tools,
                max_iters=8,
            )

            # Run the agent (use acall for async context)
            result = await agent.acall(
                question="Search for recent news about AI agents"
            )
            print(f"\nAnswer: {result.answer}")


asyncio.run(main())
```

The key function is `dspy.Tool.from_mcp_tool(session, tool)`. It takes an MCP client session and an MCP tool definition, and returns a `dspy.Tool` that you can use anywhere in DSPy. The conversion preserves:

- **Tool name**: used by the LM for tool selection
- **Tool description**: tells the LM what the tool does
- **Parameter schemas**: argument names, types, and descriptions

---

## Connecting to Local MCP Servers (Stdio)

Local MCP servers run as child processes and communicate over standard input/output. This is common for development and for servers that need access to local resources (file systems, local databases).

```python

import asyncio
import dspy
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


async def main():
    # Define the local MCP server to launch
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "./documents"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            mcp_tools = await session.list_tools()
            print(f"Found {len(mcp_tools.tools)} tools from filesystem server")

            # Convert to DSPy tools
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in mcp_tools.tools
            ]

            # Build an agent that can read and search files
            agent = dspy.ReAct(
                "question -> answer",
                tools=dspy_tools,
                max_iters=6,
            )

            result = await agent.acall(
                question="What files are in the documents directory?"
            )
            print(f"\nAnswer: {result.answer}")


asyncio.run(main())
```

The `StdioServerParameters` tell DSPy to launch the MCP server as a subprocess. The `stdio_client` context manager handles starting the process, establishing communication, and cleaning up when done.

---

## MCP + Local Tools Together

You're not limited to MCP tools alone. Combine them with local Python functions for the best of both worlds:

```python
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
```

---

## Key Takeaways

- **MCP is an open protocol** for standardized tool access. Write once, use from any compatible client.
- **`dspy.Tool.from_mcp_tool(session, tool)`** converts MCP tools to DSPy tools, preserving names, descriptions, and parameter schemas.
- **Remote servers** use `streamablehttp_client` for HTTP transport, ideal for production.
- **Local servers** use `stdio_client` with `StdioServerParameters`, ideal for development and local resources.
- **MCP tools are async**. Use `agent.acall()` in an async context to run agents with MCP tools.
- **Combine MCP and local tools** in the same agent for maximum flexibility.

---

## Next Up

Agents are powerful, but they're stateless by default. Each call starts from scratch with no memory of previous interactions. In the next post, we'll add **memory and conversation history** to our agents, enabling multi-turn conversations and persistent context.

**[6.4: Memory and Conversation History →](../6.4-memory-agents/blog.md)**

---

## Resources

- 📖 [DSPy MCP Integration](https://dspy.ai/tutorials/mcp/)
- 📖 [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- 📖 [MCP Server Registry](https://github.com/modelcontextprotocol/servers)
- 📖 [DSPy Tool API Reference](https://dspy.ai/api/tools/Tool/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
