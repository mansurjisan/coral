# Adding MCP Servers to CORAL

CORAL discovers tools from MCP (Model Context Protocol) servers at startup. Adding a new server gives the LLM new capabilities.

## 1. Add to coral_config.json

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"]
    }
  }
}
```

For Python-based servers in the CORAL codebase:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "coral.servers.my_server"]
    }
  }
}
```

## 2. Add tool routing (important for small models)

In `src/coral/agent.py`, add a routing entry to `_TOOL_ROUTES`:

```python
_TOOL_ROUTES: list[tuple[list[str], list[str]]] = [
    # ... existing routes ...
    (["my keyword", "another keyword"],
     ["my_tool_prefix_"]),
]
```

This ensures queries matching those keywords only send relevant tools to the LLM, preventing confusion when there are many tools.

## 3. Update the system prompt

In `src/coral/prompts.py`, add a tool mapping line:

```python
- "my data type" -> call my_tool_function(...)
```

## 4. Writing a custom MCP server

Use FastMCP to create a server in `src/coral/servers/`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(param: str) -> str:
    """Description of what this tool does.

    Args:
        param: Description of param.
    """
    # Implementation
    return "result"

if __name__ == "__main__":
    mcp.run()
```

## 5. Test

```bash
# Verify the server starts and tools are discovered
coral tools --config coral_config.json

# Test with the agent
coral chat --model qwen3:8b
```
