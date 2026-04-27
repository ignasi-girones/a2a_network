"""MCP Tools Server — provides static tools for Specialized Agents.

Demonstrates the complementarity of A2A + MCP:
- A2A: inter-agent communication protocol
- MCP: tool access protocol (agents access external capabilities)

This server runs independently and agents connect to it as MCP clients.
"""

import math
import re

from mcp.server.fastmcp import FastMCP

from common.config import settings

mcp = FastMCP(
    name="Debate Tools",
    instructions="Tools available to specialized debate agents for enhanced reasoning.",
    host="0.0.0.0",
    port=settings.mcp_port,
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


@mcp.tool()
async def wikipedia(title: str) -> str:
    """Fetch the lead summary of a Wikipedia article.

    Uses the public REST API (``en.wikipedia.org/api/rest_v1/page/summary``)
    — no key required. Phase 3 routing: the Analista uses this for
    encyclopedic baseline facts, complementing the Buscador's web_search.

    Args:
        title: Article title (e.g. "Microservices", "Bayesian inference").
            Spaces are URL-encoded automatically.

    Returns:
        A short paragraph (the article's "extract") plus the canonical URL,
        or an explanatory string when no article is found.
    """
    import httpx
    import urllib.parse

    safe_title = urllib.parse.quote(title.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Accept": "application/json"},
                timeout=10.0,
            )
            if response.status_code == 404:
                return f"Wikipedia: no article found for '{title}'."
            response.raise_for_status()
            data = response.json()
            extract = (data.get("extract") or "").strip()
            page_url = (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            )
            if not extract:
                return f"Wikipedia: article for '{title}' has no extract."
            out = f"Wikipedia · {data.get('title', title)}: {extract}"
            if page_url:
                out += f"\nSource: {page_url}"
            return out
    except Exception as e:
        return f"Wikipedia lookup failed: {e}."


@mcp.tool()
async def arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for paper titles + abstracts matching ``query``.

    Uses arXiv's public Atom-feed API (``export.arxiv.org/api/query``).
    Phase 3 routing: the Buscador and Sintetizador use this for academic
    grounding when the topic touches research-heavy domains.

    Args:
        query: Free-text search query (passed as ``search_query=all:<query>``).
        max_results: Cap on returned entries; clamped to [1, 5] to keep
            tokens bounded.

    Returns:
        A bullet list of the top results — each line ``Title (id) — first
        sentence of the abstract`` — or a friendly message if nothing matched.
    """
    import re as _re
    import urllib.parse

    import httpx

    n = max(1, min(5, int(max_results)))
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": n,
    }
    qs = urllib.parse.urlencode(params)
    url = f"http://export.arxiv.org/api/query?{qs}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            xml_text = response.text
    except Exception as e:
        return f"arXiv lookup failed: {e}."

    # Quick-and-dirty Atom parsing — we don't want to add a dependency.
    entries = _re.findall(
        r"<entry>(.*?)</entry>", xml_text, flags=_re.DOTALL
    )
    if not entries:
        return f"arXiv: no results for '{query}'."

    out_lines: list[str] = []
    for entry in entries[:n]:
        title_m = _re.search(r"<title>(.*?)</title>", entry, flags=_re.DOTALL)
        id_m = _re.search(r"<id>(.*?)</id>", entry)
        summary_m = _re.search(
            r"<summary>(.*?)</summary>", entry, flags=_re.DOTALL
        )
        title = (title_m.group(1).strip() if title_m else "(untitled)")
        title = _re.sub(r"\s+", " ", title)
        paper_id = (id_m.group(1).strip() if id_m else "")
        summary = (summary_m.group(1).strip() if summary_m else "")
        first_sentence = _re.split(r"(?<=[.!?])\s", summary, maxsplit=1)[0]
        first_sentence = _re.sub(r"\s+", " ", first_sentence)[:280]
        line = f"- {title}"
        if paper_id:
            line += f" ({paper_id})"
        if first_sentence:
            line += f" — {first_sentence}"
        out_lines.append(line)

    return "\n".join(out_lines)


def main():
    """Run the MCP tools server."""
    print(f"MCP Tools Server starting on http://0.0.0.0:{settings.mcp_port}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
