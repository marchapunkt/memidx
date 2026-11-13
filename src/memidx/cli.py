"""Command line interface: memidx init | update | search | get | status | serve | mcp | model list|download.

Exit codes: 0 = success / at least one hit, 1 = searched correctly but no hit (or unknown path for `get`),
2 = usage or configuration error (also: unknown zone, query empty after normalization),
3 = no usable index, git failure in a work tree, or remote server unreachable.
"""
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
    p.add_argument("--config", default=s, help="trusted config file (default: $MEMIDX_CONFIG); layered over the user "
                                               "config and <corpus>/memidx.toml")
    p.add_argument("--corpus", default=s, help="Markdown repository to index (default: $MEMIDX_CORPUS or the current directory)")
    p.add_argument("--db", default=s, help="index database (default: config `db`, $MEMIDX_DB or a per-user cache directory)")
    p.add_argument("--no-corpus-config", action="store_true", default=s,
                   help="do not read <corpus>/memidx.toml ($MEMIDX_NO_CORPUS_CONFIG=1)")
    p.add_argument("-I", "--isolated", action="store_true", default=s,
                   help="ignore all MEMIDX_* environment variables and the user config")
    p.add_argument("--trust-repo-owner", action="store_true", default=s,
                   help="let git read a repository owned by another user (git_safe_directory)")
    return p


def build_parser():
    common = _common()
    ap = argparse.ArgumentParser(prog="memidx", parents=[common],
                                 description="Section-level hybrid search over the Markdown files of a git repository.")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("init", parents=[common], help="write a commented example config")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--user", action="store_true", help="write the full example to the user config file")

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
    p.add_argument("--client", default=None, help="self-declared client id ($MEMIDX_CLIENT)")
    p.add_argument("--url", default=None, help="query a running server at this URL instead of the local index")

    p = sub.add_parser("get", parents=[common], help="print a file (or line range) from the index: PATH[:FROM-TO]")
    p.add_argument("spec")
    p.add_argument("--client", default=None)

    sub.add_parser("status", parents=[common], help="show index statistics and where the configuration came from")

    p = sub.add_parser("serve", parents=[common], help="HTTP API + read UI + auto-update")
    p.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="HTTP port (default 8800)")
    p.add_argument("--interval", type=int, default=None, help="auto-update interval in seconds (default 600, 0 = only at start)")
    p.add_argument("--no-auto-update", action="store_true", help="do not index; serve the existing index")
    p.add_argument("--no-embed", action="store_true", help="skip vectors when updating")
    p.add_argument("--mcp", action="store_true", help="also run the MCP server (streamable HTTP, mcp_host:mcp_port)")

    p = sub.add_parser("mcp", parents=[common], help="run the MCP server")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default=None,
                   help="default stdio ($MEMIDX_MCP_TRANSPORT)")
    p.add_argument("--host", default=None, help="bind address for streamable-http (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port for streamable-http (default 8801)")

    p = sub.add_parser("model", parents=[common], help="list or download embedding model presets")
    msub = p.add_subparsers(dest="model_cmd", metavar="ACTION")
    msub.required = True
    msub.add_parser("list", parents=[common], help="show the model presets and their licenses")
    d = msub.add_parser("download", parents=[common], help="download one model preset into the model cache")
    d.add_argument("preset", help="preset name (see `memidx model list`)")
    d.add_argument("--accept-license", action="store_true",
                   help="required for a preset whose license is not OSI-approved")
    d.add_argument("--cache-dir", default=None, help="model cache (default: config model_cache or ~/.cache/memidx/models)")
    return ap


def _err(msg):
    print(f"memidx: {msg}", file=sys.stderr, flush=True)


def _env(args):
    """os.environ, or without MEMIDX_* under --isolated."""
    env = dict(os.environ)
    return config_mod.strip_memidx_env(env) if getattr(args, "isolated", False) else env


def _cfg(args, vectors_check=False):
    cfg = config_mod.load_config(getattr(args, "config", None), getattr(args, "corpus", None), getattr(args, "db", None),
                                 isolated=getattr(args, "isolated", False),
                                 corpus_config=False if getattr(args, "no_corpus_config", False) else None,
                                 trust_repo_owner=getattr(args, "trust_repo_owner", False))
    for n in cfg.notices:
        _err(n)
    if vectors_check:
        config_mod.check_vectors(cfg)
    return core.configure(cfg)


def _location_notice(cfg):
    names = cfg.location_env()
    if names:
        _err(f"environment variable(s) in effect that change where data is read or written: {', '.join(names)} "
             "(use --isolated to ignore MEMIDX_* variables)")


def cmd_init(args):
    env = _env(args)
    if args.user:
        target = config_mod.user_config_path(env)
        text = config_mod.EXAMPLE
    else:
        corpus = Path(getattr(args, "corpus", None) or env.get("MEMIDX_CORPUS") or os.getcwd())
        target = Path(getattr(args, "config", None) or corpus / config_mod.CONFIG_NAME)
        inside = config_mod.is_inside(target.parent if not target.exists() else target, corpus)
        text = config_mod.example_corpus(config_mod.user_config_path(env)) if inside else config_mod.EXAMPLE
    if target.exists() and not args.force:
        _err(f"{target} exists (use --force to overwrite)")
        return 1
    core.secure_makedirs(str(target.parent))
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return 0


def cmd_update(args):
    cfg = _cfg(args, vectors_check=True)
    _location_notice(cfg)
    core.update(cfg.corpus, cfg.db, embed=cfg.vectors and not args.no_embed)
    return 0


def _client(args):
    if args.client is not None:
        return args.client
    return _env(args).get("MEMIDX_CLIENT")


def _print_result(args, res):
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(core.render_list(res) if args.list else core.render_text(res))
    if res.get("private_excluded"):
        _err("private paths were not searched")
    reason = res.get("reason")
    if res.get("hits"):
        return 0
    if reason in ("empty_query", "unknown_zone"):
        _err(res.get("message", reason))
    return core.EXIT_CODES.get(reason or "no_match", 1)


def _http_search(args, q):
    params = {"q": q, "k": args.k, "budget": args.budget, "zone": args.zone or "", "mode": "json",
              "all": "1" if args.all else "", "client": _client(args) or ""}
    url = args.url.rstrip("/") + "/search?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            res = json.loads(body)
        except ValueError:
            _err(f"server error {e.code}: {body[:200]}")
            return 2 if 400 <= e.code < 500 else 3
        _err(f"server answered {e.code}: {res.get('message', '')}")
        return core.EXIT_CODES.get(res.get("reason"), 2 if 400 <= e.code < 500 else 3)
    except OSError as e:
        _err(f"server not reachable ({e})")
        return 3
    try:
        res = json.loads(body)
    except ValueError:
        _err(f"unexpected answer from the server: {body[:200]}")
        return 3
    return _print_result(args, res)


def cmd_search(args):
    q = " ".join(args.query)
    if args.url:
        return _http_search(args, q)
    cfg = _cfg(args, vectors_check=True)
    _location_notice(cfg)
    db = core.open_index(cfg.db)
    try:
        res = core.search(db, q, args.k, args.budget, args.zone, client=_client(args), include_invalid=args.all)
    finally:
        db.close()
    return _print_result(args, res)


def cmd_get(args):
    cfg = _cfg(args)
    path, _, rng = args.spec.partition(":")
    lo, _, hi = rng.partition("-")
    try:
        lo, hi = int(lo or 1), int(hi or 0)
    except ValueError:
        raise ValueError(f"bad line range in {args.spec!r}; use PATH[:FROM-TO]")
    db = core.open_index(cfg.db)
    try:
        txt = core.get_lines(db, path, lo, hi, client=_client(args))
    finally:
        db.close()
    if txt is None:
        _err("unknown path")
        return 1
    print(txt)
    return 0


def cmd_status(args):
    cfg = _cfg(args)
    env_set = sorted(k for k in os.environ if k.startswith("MEMIDX_"))
    effective = sorted(cfg.env_used)
    st = {"corpus": cfg.corpus, "db": cfg.db,
          "config": [{"path": p, "trust": "untrusted (limited keys)" if t == "corpus" else "trusted", "kind": t}
                     for p, t in cfg.sources],
          "isolated": cfg.isolated, "corpus_config": cfg.corpus_config,
          "environment": {"set": env_set, "effective": effective,
                          "ignored": [n for n in env_set if n not in effective]},
          "languages": list(cfg.languages), "vectors_enabled": cfg.vectors, "model": cfg.model or None,
          "vector_extra_installed": embed.deps_available(), "git_safe_directory": cfg.git_safe_directory}
    problem = core.index_problem(cfg.db)
    if problem:
        st["index"] = {"usable": False, "reason": "no_index", "detail": problem[0], "message": problem[1]}
        print(json.dumps(st, ensure_ascii=False, indent=2))
        _err(problem[1])
        return 3
    db = core.open_db(cfg.db, readonly=True)
    try:
        st.update(core.status(db))
    finally:
        db.close()
    st["index"] = {"usable": True}
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args):
    from . import server
    cfg = _cfg(args, vectors_check=True)
    _location_notice(cfg)
    server.serve(cfg, host=args.host, port=args.port, interval=args.interval,
                 embed=cfg.vectors and not args.no_embed, auto=not args.no_auto_update, with_mcp=args.mcp)
    return 0


def cmd_mcp(args):
    from . import mcp_server
    cfg = _cfg(args, vectors_check=True)
    p = core.index_problem(cfg.db)
    if p:
        raise core.NoIndex(*p)
    transport = args.transport or _env(args).get("MEMIDX_MCP_TRANSPORT") or "stdio"
    if transport not in ("stdio", "streamable-http"):
        raise ValueError(f"unknown MCP transport {transport!r}")
    mcp_server.run(cfg, transport, args.host, args.port)
    return 0


def cmd_model(args):
    if args.model_cmd == "list":
        for p in embed.PRESETS.values():
            print(f"{p.name}\n  repository:   {p.repository}@{p.revision[:12]} ({p.download_size})\n"
                  f"  license:      {p.license_name} ({p.license_url})\n"
                  f"  OSI-approved: {'yes' if p.osi_approved else 'NO'}")
        print("memidx ships no model; full-text search needs none. Vectors are opt-in (vectors = true, model = ...).")
        return 0
    preset = embed.get_preset(args.preset)          # ValueError -> exit 2
    if not preset.osi_approved and not args.accept_license:
        _err(f"refusing to download {preset.name}: it is distributed under {preset.license_name} "
             f"({preset.license_url}), which is NOT an OSI-approved open-source license. Read the terms; if you "
             "accept them, run the command again with --accept-license.")
        return 2
    cfg = _cfg(args)
    cache = args.cache_dir or cfg.model_cache
    try:
        embed.download_model(cache, preset)
    except embed.VectorsUnavailable as e:
        _err(str(e))
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
    except (core.NoIndex, core.GitError) as e:
        _err(str(e))
        return 3
    except ValueError as e:          # configuration and usage errors
        _err(str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
