"""MCP Tools Server — provides static tools for Specialized Agents.

Demonstrates the complementarity of A2A + MCP:
- A2A: inter-agent communication protocol
- MCP: tool access protocol (agents access external capabilities)

This server runs independently and agents connect to it as MCP clients.
"""

import math
import re

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Debate Tools",
    instructions="Tools available to specialized debate agents for enhanced reasoning.",
    host="0.0.0.0",
    port=8100,
)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic (+, -, *, /, **), parentheses,
    and common functions (sqrt, abs, round, min, max, log, pow).

    Args:
        expression: A mathematical expression to evaluate (e.g. "2 * 3 + 1", "sqrt(144)")

    Returns:
        The result as a string, or an error message.
    """
    # Sanitize: only allow numbers, operators, parentheses, and safe functions
    safe_pattern = r'^[\d\s\+\-\*/\.\(\)\,\%]+$'
    func_pattern = r'\b(sqrt|abs|round|min|max|log|pow|int|float)\b'

    clean_expr = re.sub(func_pattern, lambda m: f"__{m.group()}__", expression)
    clean_expr = re.sub(r'__(\w+)__', lambda m: m.group(1), clean_expr)

    # Build safe namespace
    safe_ns = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "log": math.log,
        "pow": pow,
        "int": int,
        "float": float,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Extra safety check: no builtins, no dunder access
        if "__" in expression or "import" in expression:
            return "Error: forbidden expression"
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web for information relevant to the debate.

    Uses DuckDuckGo Instant Answer API (free, no API key needed).

    Args:
        query: The search query string.

    Returns:
        A summary of search results, or an indication that no results were found.
    """
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10.0,
            )
            data = response.json()

            results = []

            # Abstract (main answer)
            if data.get("Abstract"):
                results.append(f"Summary: {data['Abstract']}")
                if data.get("AbstractSource"):
                    results.append(f"Source: {data['AbstractSource']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")

            if not results:
                return f"No instant results found for '{query}'. The agents should rely on their training knowledge."

            return "\n".join(results)

    except Exception as e:
        return f"Search failed: {e}. The agents should rely on their training knowledge."


def main():
    """Run the MCP tools server."""
    print("MCP Tools Server starting on http://0.0.0.0:8100")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
