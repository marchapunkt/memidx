"""Private prefixes: only clients whose self-declared id starts with own_client_prefix see them
(ported from private_test.py)."""
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from memidx import core, server

SECRET = "quokkahatsecret"          # only in the private file
PUBLIC = "zebrapublicword"
PRIV = "private/notes.md"
OWN = "own:"


@pytest.fixture
def env(idx, tmp_path):
    idx.build(docs={
        "notes/open.md": ("Open note", "Public description", f"{PUBLIC} is in the public note about the service."),
        "notes/second.md": ("Second note", "Another one", f"{PUBLIC} and the service run."),
        PRIV: ("Private note", "Private description", f"{SECRET} is only in the private note. {PUBLIC} too."),
    }, private_prefixes=("private/",), own_client_prefix=OWN, log_path=str(tmp_path / "queries.jsonl"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(idx.dbpath))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def http(path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")
    yield idx, idx.db(), http, port
    srv.shutdown(); srv.server_close()


def paths(res):
    return [t["path"] for t in res["hits"]]


def test_sees_private(env):
    for c in ("own:laptop", "own:"):
        assert core.sees_private(c)
    for c in (None, "", "cli", "http", "mcp", "Own:x", "xown:y"):
        assert not core.sees_private(c)


def test_secret_word_only_with_own_id(env):
    _, db, _, _ = env
    for c in (None, "cli", "http", "mcp"):
        assert core.search(db, SECRET, client=c)["hits"] == []
    assert paths(core.search(db, SECRET, client="own:laptop")) == [PRIV]


def test_shared_word_without_own_id_only_public(env):
    _, db, _, _ = env
    p = paths(core.search(db, PUBLIC, k=10, client="worker"))
    assert p and all(not x.startswith("private/") for x in p)
    assert PRIV in paths(core.search(db, PUBLIC, k=10, client="own:office"))


def test_zone_filter_without_own_id(env):
    _, db, _, _ = env
    p = paths(core.search(db, PUBLIC, k=10, zone="default", client="mcp"))
    assert p and all(not x.startswith("private/") for x in p)


def test_get_lines_and_related(env):
    _, db, _, _ = env
    assert core.get_lines(db, PRIV, client="worker") is None
    assert core.get_lines(db, PRIV) is None
    assert SECRET in core.get_lines(db, PRIV, client="own:x")
    assert PUBLIC in core.get_lines(db, "notes/open.md", client="worker")
    assert core.related(db, PRIV, client="mcp") == []
    assert all(not t["path"].startswith("private/") for t in core.related(db, "notes/open.md", client="mcp"))
    assert PRIV in [t["path"] for t in core.related(db, "notes/open.md", k=5, client="own:x")]


def test_graph_cache_separated_by_visibility(env):
    _, db, _, _ = env
    other = core.graph(db, 300, client="worker")
    own = core.graph(db, 300, client="own:x")
    none = core.graph(db, 300, client=None)
    assert PRIV not in [n["path"] for n in other["nodes"]]
    assert PRIV in [n["path"] for n in own["nodes"]]
    assert len(other["nodes"]) + 1 == len(own["nodes"])
    assert PRIV not in [n["path"] for n in none["nodes"]]
    assert all(not p.startswith("private/") for e in other["edges"] for p in e[:2])


def test_mode_all_shows_everything(env):
    idx, db, _, _ = env
    idx.configure(private_prefixes=("private/",), private_mode="all")
    assert paths(core.search(db, SECRET, client="worker")) == [PRIV]
    assert core.get_lines(db, PRIV) is not None


def test_empty_prefixes_nothing_private(env):
    idx, db, _, _ = env
    idx.configure()
    assert paths(core.search(db, SECRET, client=None)) == [PRIV]


def test_own_prefix_configurable(env):
    idx, _, _, _ = env
    idx.configure(private_prefixes=("private/",), own_client_prefix="me-")
    assert core.sees_private("me-laptop") and not core.sees_private("own:laptop")


def test_http_endpoints(env):
    _, _, http, _ = env
    q, pf = urllib.parse.quote(SECRET), urllib.parse.quote(PRIV)
    for kl in ("", "&client=worker", "&client=mcp"):
        assert json.loads(http(f"/search?q={q}{kl}")[1])["hits"] == [], kl
        assert http(f"/doc?path={pf}{kl}")[0] == 404, kl
        assert json.loads(http(f"/related?path={pf}{kl}")[1]) == [], kl
        assert PRIV not in [d["path"] for d in json.loads(http(f"/list?limit=100{kl}")[1])["files"]]
        assert PRIV not in [n["path"] for n in json.loads(http(f"/graph?limit=100{kl}")[1])["nodes"]]
        assert "notes.md" not in http(f"/?q={q}{kl}")[1]
    kl = "&client=own:laptop"
    assert [t["path"] for t in json.loads(http(f"/search?q={q}{kl}")[1])["hits"]] == [PRIV]
    assert http(f"/doc?path={pf}{kl}")[0] == 200
    assert PRIV in [d["path"] for d in json.loads(http(f"/list?limit=100{kl}")[1])["files"]]
    assert PRIV in [n["path"] for n in json.loads(http(f"/graph?limit=100{kl}")[1])["nodes"]]


def test_query_log_records_visibility(env, tmp_path):
    _, db, _, _ = env
    core.search(db, SECRET, client="worker")
    core.search(db, SECRET, client="own:x")
    core.search(db, SECRET)                       # no client: not logged
    rows = [json.loads(l) for l in open(tmp_path / "queries.jsonl", encoding="utf-8")]
    assert [(r["client"], r["private"], r["n"]) for r in rows] == [("worker", False, 0), ("own:x", True, 1)]


def test_no_log_by_default(idx, tmp_path):
    db = idx.build({"a.md": "# A\n\nword\n"})
    core.search(db, "word", client="x")
    assert not list(tmp_path.glob("*.jsonl"))


def test_cli_sends_client_to_server(env):
    from memidx import cli
    _, _, _, port = env
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.main(["search", SECRET, "--url", f"http://127.0.0.1:{port}", "--client", "own:cli", "--list"])
    assert rc == 0 and PRIV in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(["search", SECRET, "--url", f"http://127.0.0.1:{port}", "--list"])
    assert "(no hits)" in out.getvalue()
