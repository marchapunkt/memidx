"""MCP server (stdio or streamable HTTP). Tools: memory_search, memory_get, memory_related, memory_status.

Needs the extra `memidx[mcp]`. Works with the `mcp` package 2.x (MCPServer) and 1.x (FastMCP).
An MCP server cannot tell its callers apart: every call uses the client id `mcp_client` from the
configuration (default "mcp", which does not see private paths).
"""
import json

from . import core

INSTRUCTIONS = ("Search over a Markdown knowledge base. Call memory_search first, then memory_get for a line range "
                "if you need more. Answers contain only the matching sections with their source (path, heading, "
                "lines); configured hints (for example about renamed or retired things) come first.")


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer as server   # mcp 2.x
        return server, False
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP as server       # mcp 1.x
        return server, True
    except ImportError as e:
        raise SystemExit("the MCP server needs the extra: pip install 'memidx[mcp]'") from e


def build_server(cfg, host=None, port=None):
    """Creates the MCP server object with its four tools (each call opens its own database connection)."""
    server_cls, v1 = _server_class()
    host = host or cfg.mcp_host
    port = cfg.mcp_port if port is None else port
    mcp = server_cls("memidx", host=host, port=port, instructions=INSTRUCTIONS) if v1 else \
        server_cls("memidx", instructions=INSTRUCTIONS)
    client = cfg.mcp_client

    def with_db(fn):
        p = core.index_problem(cfg.db)               # read-only; never creates a database
        if p:
            return f"(no usable index) [no_index] {p[1]}"
        db = core.open_db(cfg.db, readonly=True)
        try:
            return fn(db)
        finally:
            db.close()

    def search_text(db, query, k, budget_tokens, zone):
        res = core.search(db, query, max(1, min(k, 10)), max(200, min(budget_tokens, 4000)), zone, client=client)
        text = core.render_text(res)
        if res.get("private_excluded"):
            text += "\n\n(private paths were not searched)"
        return text

    @mcp.tool()
    def memory_search(query: str, k: int = 3, budget_tokens: int = 800, zone: str | None = None) -> str:
        """Searches the whole knowledge base and returns only the matching SECTIONS (source path, heading,
        lines) within a token budget. `zone` optionally limits the search to one configured zone."""
        return with_db(lambda db: search_text(db, query, k, budget_tokens, zone))

    @mcp.tool()
    def memory_get(path: str, from_line: int = 1, to_line: int = 0) -> str:
        """Returns the text of one file (only the sections in the line range; to_line=0 means to the end,
        truncated to about 12000 characters)."""
        return with_db(lambda db: core.get_lines(db, path, from_line, to_line, client=client) or "unknown path")

    @mcp.tool()
    def memory_related(path: str, k: int = 5) -> str:
        """Related sections of other files: wikilink neighbours first, then text similarity."""
        return with_db(lambda db: "\n\n".join(f'== {t["path"]} :: {t["section"]} (lines {t["lines"]})'
                                              for t in core.related(db, path, k, client=client)) or "(none)")

    @mcp.tool()
    def memory_status() -> str:
        """Index state: files, sections, vectors, zones."""
        return with_db(lambda db: json.dumps(core.status(db), ensure_ascii=False))

    mcp._memidx_v1 = v1
    return mcp


def run(cfg, transport="stdio", host=None, port=None):
    mcp = build_server(cfg, host, port)
    if transport == "stdio" or mcp._memidx_v1:
        mcp.run(transport=transport)
    else:
        mcp.run(transport=transport, host=host or cfg.mcp_host, port=cfg.mcp_port if port is None else port)
