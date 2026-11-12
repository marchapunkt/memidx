"""Command line interface: memidx init | update | search | get | status | serve | mcp | model download."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config as config_mod
from . import core, embed


def _common():
    p = argparse.ArgumentParser(add_help=False)
    s = argparse.SUPPRESS
    p.add_argument("--config", default=s, help="config file (default: $MEMIDX_CONFIG, then <corpus>/memidx.toml)")
    p.add_argument("--corpus", default=s, help="Markdown repository to index (default: $MEMIDX_CORPUS or the current directory)")
    p.add_argument("--db", default=s, help="index database (default: config `db`, $MEMIDX_DB or <corpus>/.memidx/index.db)")
    return p


def build_parser():
    common = _common()
    ap = argparse.ArgumentParser(prog="memidx", parents=[common],
                                 description="Section-level hybrid search over the Markdown files of a git repository.")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("init", parents=[common], help="write a commented example memidx.toml")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")

    p = sub.add_parser("update", parents=[common], help="build or update the index (incremental)")
    p.add_argument("--no-embed", action="store_true", help="skip vectors even when they are available")

    p = sub.add_parser("search", parents=[common], help="search the index")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", "--top", dest="k", type=int, default=3, help="number of hits (default 3)")
    p.add_argument("--budget", type=int, default=800, help="token budget for all hits (default 800)")
    p.add_argument("--zone", default=None, help="only this zone")
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--list", action="store_true", help="one line per hit")
    fmt.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--all", action="store_true", help="also show invalid memory entries")
    p.add_argument("--client", default=os.environ.get("MEMIDX_CLIENT"), help="self-declared client id ($MEMIDX_CLIENT)")
    p.add_argument("--url", default=os.environ.get("MEMIDX_URL"), help="query a running server instead of the local index ($MEMIDX_URL)")

    p = sub.add_parser("get", parents=[common], help="print a file (or line range) from the index: PATH[:FROM-TO]")
    p.add_argument("spec")
    p.add_argument("--client", default=os.environ.get("MEMIDX_CLIENT"))

    p = sub.add_parser("status", parents=[common], help="show index statistics")

    p = sub.add_parser("serve", parents=[common], help="HTTP API + read UI + auto-update")
    p.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="HTTP port (default 8800)")
    p.add_argument("--interval", type=int, default=None, help="auto-update interval in seconds (default 600, 0 = only at start)")
    p.add_argument("--no-auto-update", action="store_true", help="do not index; serve the existing index")
    p.add_argument("--no-embed", action="store_true", help="skip vectors when updating")
    p.add_argument("--mcp", action="store_true", help="also run the MCP server (streamable HTTP, mcp_host:mcp_port)")

    p = sub.add_parser("mcp", parents=[common], help="run the MCP server")
    p.add_argument("--transport", choices=["stdio", "streamable-http"],
                   default=os.environ.get("MEMIDX_MCP_TRANSPORT", "stdio"))
    p.add_argument("--host", default=None, help="bind address for streamable-http (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port for streamable-http (default 8801)")

    p = sub.add_parser("model", parents=[common], help="manage the embedding model")
    msub = p.add_subparsers(dest="model_cmd", metavar="ACTION")
    msub.required = True
    d = msub.add_parser("download", parents=[common], help="download EmbeddingGemma-300m (ONNX) into the model cache")
    d.add_argument("--cache-dir", default=None, help="model cache (default: config model_cache or ~/.cache/memidx/models)")
    return ap


def _cfg(args):
    cfg = config_mod.load_config(getattr(args, "config", None), getattr(args, "corpus", None), getattr(args, "db", None))
    return core.configure(cfg)


def _need_index(cfg):
    if not os.path.exists(cfg.db):
        print(f"memidx: no index at {cfg.db}; run `memidx update` first", file=sys.stderr)
        raise SystemExit(2)


def cmd_init(args):
    corpus = Path(getattr(args, "corpus", None) or os.environ.get("MEMIDX_CORPUS") or os.getcwd())
    target = Path(getattr(args, "config", None) or corpus / config_mod.CONFIG_NAME)
    if target.exists() and not args.force:
        print(f"memidx: {target} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config_mod.EXAMPLE, encoding="utf-8")
    print(f"wrote {target}")
    return 0


def cmd_update(args):
    cfg = _cfg(args)
    core.update(cfg.corpus, cfg.db, embed=cfg.vectors and not args.no_embed)
    return 0


def _http_search(args, q):
    mode = "list" if args.list else ("json" if args.json else "text")
    params = {"q": q, "k": args.k, "budget": args.budget, "zone": args.zone or "", "mode": mode,
              "all": "1" if args.all else "", "client": args.client or ""}
    url = args.url.rstrip("/") + "/search?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            print(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"memidx: server error {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"memidx: server not reachable ({e})", file=sys.stderr)
        return 3
    return 0


def cmd_search(args):
    q = " ".join(args.query)
    if args.url:
        return _http_search(args, q)
    cfg = _cfg(args)
    _need_index(cfg)
    db = core.open_db(cfg.db)
    try:
        res = core.search(db, q, args.k, args.budget, args.zone, client=args.client, include_invalid=args.all)
    finally:
        db.close()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(core.render_list(res) if args.list else core.render_text(res))
    return 0


def cmd_get(args):
    cfg = _cfg(args)
    _need_index(cfg)
    path, _, rng = args.spec.partition(":")
    lo, _, hi = rng.partition("-")
    db = core.open_db(cfg.db)
    try:
        txt = core.get_lines(db, path, int(lo or 1), int(hi or 0), client=args.client)
    finally:
        db.close()
    if txt is None:
        print("unknown path", file=sys.stderr)
        return 1
    print(txt)
    return 0


def cmd_status(args):
    cfg = _cfg(args)
    _need_index(cfg)
    db = core.open_db(cfg.db)
    try:
        st = core.status(db)
    finally:
        db.close()
    st = {"corpus": cfg.corpus, "db": cfg.db, "config": cfg.source or "(defaults)",
          "languages": list(cfg.languages), **st,
          "vectors_enabled": cfg.vectors, "vector_extra_installed": embed.deps_available()}
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args):
    from . import server
    cfg = _cfg(args)
    server.serve(cfg, host=args.host, port=args.port, interval=args.interval,
                 embed=cfg.vectors and not args.no_embed, auto=not args.no_auto_update, with_mcp=args.mcp)
    return 0


def cmd_mcp(args):
    from . import mcp_server
    cfg = _cfg(args)
    _need_index(cfg)
    mcp_server.run(cfg, args.transport, args.host, args.port)
    return 0


def cmd_model(args):
    cfg = _cfg(args)
    cache = args.cache_dir or cfg.model_cache
    try:
        embed.download_model(cache)
    except embed.VectorsUnavailable as e:
        print(f"memidx: {e}", file=sys.stderr)
        return 1
    return 0


COMMANDS = {"init": cmd_init, "update": cmd_update, "search": cmd_search, "get": cmd_get, "status": cmd_status,
            "serve": cmd_serve, "mcp": cmd_mcp, "model": cmd_model}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](args) or 0
    except ValueError as e:          # configuration errors
        print(f"memidx: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
