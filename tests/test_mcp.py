"""MCP tools (needs the `mcp` extra; skipped otherwise). Calls the tools in-process, no transport."""
import asyncio
import json

import pytest

pytest.importorskip("mcp")

from memidx import mcp_server  # noqa: E402


def _text(result):
    """Normalizes the call_tool result shapes of mcp 1.x and 2.x to the returned string."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        result = result.content
    if isinstance(result, list):
        return "".join(getattr(c, "text", "") for c in result)
    return str(result)


def test_tools(idx):
    idx.build(docs={"notes/a.md": ("Alpha", "first", "The quokka lives on an island."),
                    "private/p.md": ("Private", "secret", "quokka diary")},
              private_prefixes=("private/",), vectors=False)
    mcp = mcp_server.build_server(idx.cfg)

    async def go():
        names = {t.name for t in await mcp.list_tools()}
        assert names == {"memory_search", "memory_get", "memory_related", "memory_status"}
        out = _text(await mcp.call_tool("memory_search", {"query": "quokka"}))
        assert "notes/a.md" in out and "private/p.md" not in out          # default client "mcp" sees no private paths
        out = _text(await mcp.call_tool("memory_get", {"path": "notes/a.md"}))
        assert "island" in out
        assert _text(await mcp.call_tool("memory_get", {"path": "private/p.md"})) == "unknown path"
        st = json.loads(_text(await mcp.call_tool("memory_status", {})))
        assert st["files"] == 2
    asyncio.run(go())
