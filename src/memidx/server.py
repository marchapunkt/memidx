"""HTTP API, read-only web UI and the auto-update loop."""
import html
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from . import core

PAGE = """<!doctype html><meta charset=utf-8><title>memidx</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:15px/1.5 system-ui;max-width:920px;margin:2rem auto;padding:0 1rem}input{width:70%;padding:.5rem}
.hit{border:1px solid #8884;border-radius:6px;padding:.6rem 1rem;margin:.8rem 0}.src{font:12px monospace;color:#888}
pre{white-space:pre-wrap}.hint{background:#fc03;padding:.5rem 1rem;border-radius:6px}a{color:inherit}</style>
<h2>memidx search</h2><form><input name=q value="{q}" autofocus> <button>Search</button></form>{body}"""


def _truthy(v):
    return v in ("1", "true")


def make_handler(db):
    """`db` is either an open connection (shared, as in the original) or a database path. With a path, every
    request uses its own short-lived connection, which avoids the sporadic errors of a shared connection
    under parallel requests."""
    shared = db if isinstance(db, sqlite3.Connection) else None
    dbpath = None if shared is not None else str(db)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, data):
            b = data.encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _json(self, obj, code=200):
            self._send(code, "application/json", json.dumps(obj, ensure_ascii=False))

        def _no_index(self, detail, message):
            self._json({"reason": "no_index", "detail": detail, "message": message}, 503)

        def do_GET(self):
            if shared is None:
                p = core.index_problem(dbpath)
                if p:
                    self._no_index(*p)
                    return
                conn = core.open_db(dbpath, readonly=True)       # never creates a database
            else:
                conn = shared
                p = core.connection_problem(conn)
                if p:
                    self._no_index(*p)
                    return
            try:
                self._route(conn)
            except ValueError as e:
                self._send(400, "text/plain", f"bad request: {e}")
            except Exception as e:  # errors are visible, the service stays up
                self._send(500, "text/plain", f"error: {e}")
            finally:
                if shared is None:
                    conn.close()

        def _route(self, db):
            u = urlparse(self.path); qs = parse_qs(u.query)
            g = lambda k, d="": (qs.get(k) or [d])[0]
            include_invalid = _truthy(g("all")) or _truthy(g("alle"))   # "alle" = name in the original service
            if u.path == "/search":
                res = core.search(db, g("q"), int(g("k", "3")), int(g("budget", "800")), g("zone") or None,
                                  client=g("client", "http"), include_invalid=include_invalid)
                code = 400 if res.get("reason") in ("empty_query", "unknown_zone") else 200
                if g("mode") == "list":
                    self._send(code, "text/plain", core.render_list(res))
                elif g("mode") == "text":
                    self._send(code, "text/plain", core.render_text(res))
                else:
                    self._json(res, code)
            elif u.path == "/doc":
                txt = core.get_lines(db, g("path"), int(g("from", "1")), int(g("to", "0")), client=g("client") or None)
                self._send(200 if txt else 404, "text/plain", txt or "unknown path")
            elif u.path == "/related":
                self._json(core.related(db, g("path"), client=g("client") or None, include_invalid=include_invalid))
            elif u.path == "/status":
                self._json(core.status(db))
            elif u.path == "/list":
                self._json(core.list_files(db, g("zone"), g("prefix"), int(g("limit", "200")), client=g("client") or None))
            elif u.path == "/graph":
                try:
                    lim = int(g("limit", "300"))
                except ValueError:
                    self._send(400, "text/plain", "limit must be a number"); return
                self._json(core.graph(db, lim, g("zone") or None, with_score=_truthy(g("score")), client=g("client") or None))
            elif u.path == "/":
                body, q = "", g("q")
                if q:
                    res = core.search(db, q, 5, 1500, client=None)
                    body = "".join(f'<p class=hint>{html.escape(h)}</p>' for h in res["hints"])
                    for t in res["hits"]:
                        lnk = "/doc?path=" + quote(t["path"])
                        body += (f'<div class=hit><div class=src><a href="{html.escape(lnk)}">{html.escape(t["path"])}</a> · '
                                 f'{html.escape(t["section"])} · lines {t["lines"]} · {html.escape(t["zone"])}</div>'
                                 f'<pre>{html.escape(t["text"])}</pre></div>')
                    if not res["hits"]:
                        body += f"<p>{html.escape(core.render_text({**res, 'hints': []}))}</p>"
                self._send(200, "text/html", PAGE.replace("{q}", html.escape(q)).replace("{body}", body))
            else:
                self._send(404, "text/plain", "not found")
    return H


def run_update(corpus, dbpath, embed, prefix=""):
    with core._LOCK:
        return core.update(corpus, dbpath, embed=embed, log=lambda m: print(prefix + m, flush=True))


def auto_update(corpus, dbpath, interval, embed, stop=None):
    stop = stop or threading.Event()
    while not stop.wait(interval):
        try:
            r = run_update(corpus, dbpath, embed, "[auto-update] ")
            if r["changed"] or r["removed"]:
                core.warm_graph(dbpath)
        except Exception as e:
            print("[auto-update] error:", e, flush=True)


def _mcp_child(cfg):
    args = [sys.executable, "-m", "memidx.cli", "--corpus", cfg.corpus, "--db", cfg.db]
    if cfg.explicit_config:
        args += ["--config", cfg.explicit_config]
    if cfg.isolated:
        args += ["--isolated"]
    if not cfg.corpus_config:
        args += ["--no-corpus-config"]
    args += ["mcp", "--transport", "streamable-http", "--host", cfg.mcp_host, "--port", str(cfg.mcp_port)]
    return subprocess.Popen(args)


def serve(cfg, host=None, port=None, interval=None, embed=False, auto=True, with_mcp=False, ready=None):
    """Runs the HTTP server until SIGTERM/SIGINT. With `auto`, indexes once at start and then every `interval`
    seconds. With `with_mcp`, also runs the MCP server (streamable HTTP) as a child process."""
    host = host or cfg.host
    port = cfg.port if port is None else port
    interval = cfg.interval if interval is None else interval
    stop = threading.Event()
    if auto:
        run_update(cfg.corpus, cfg.db, embed)       # an update: the only path that may create the index
    else:
        p = core.index_problem(cfg.db)
        if p:
            raise core.NoIndex(*p)
    threading.Thread(target=core.warm_graph, args=(cfg.db,), daemon=True).start()
    if auto and interval > 0:
        threading.Thread(target=auto_update, args=(cfg.corpus, cfg.db, interval, embed, stop), daemon=True).start()
    child = _mcp_child(cfg) if with_mcp else None
    httpd = ThreadingHTTPServer((host, port), make_handler(cfg.db))
    httpd.daemon_threads = True

    def _term(signum, frame):
        raise KeyboardInterrupt
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _term)   # Python ignores SIGTERM as PID 1 without a handler
    db = core.open_db(cfg.db, readonly=True)
    try:
        st = core.status(db)
    finally:
        db.close()
    print(f"memidx serving http://{host}:{httpd.server_address[1]} ({json.dumps(st)})"
          + (f", MCP on {cfg.mcp_host}:{cfg.mcp_port}" if child else ""), flush=True)
    if ready is not None:
        ready(httpd)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        if child is not None:
            child.terminate()
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill()
        print("memidx stopped", flush=True)
