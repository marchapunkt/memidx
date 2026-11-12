"""GET /graph and core.graph (ported from graph_test.py)."""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from memidx import core, server
from memidx.config import ZoneRule
from conftest import write_doc

TOPICS = ["gateway", "wissen", "graph", "board", "planer", "shell", "memory", "index", "backup", "monitoring"]
ZONES = (ZoneRule("longterm", 1.0, prefix="memory/"), ZoneRule("docs", 0.9, prefix="docs/"),
         ZoneRule("midterm", 0.85, prefix="projects/"))


@pytest.fixture
def env(idx):
    docs = {}
    for i, t in enumerate(TOPICS):
        nb = TOPICS[(i + 1) % len(TOPICS)]
        docs[f"memory/{t}.md"] = (f"Thema {t}", f"Alles zu {t} und {nb}",
                                  f"{t} {t} ist wichtig. Verweis auf {nb}. Betrieb von {t} laeuft ueber den Dienst.")
        docs[f"projects/p{i}/plan.md"] = (f"Plan {t}", f"Projektplan fuer {t}",
                                          f"Der Plan fuer {t} baut auf {nb} auf. Meilensteine fuer {t}.")
    docs["docs/alone.md"] = ("Zebrastreifen", "Voellig anderes Thema", "Quokka Zebrastreifen Xylophon.")
    idx.build(docs=docs, zones=ZONES)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(idx.dbpath))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def http(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    yield idx, http
    srv.shutdown(); srv.server_close()


def test_structure(env):
    _, http = env
    _, g = http("/graph?limit=300")
    assert g["n"] == len(g["nodes"]) == 21
    for n in g["nodes"]:
        assert set(n) == {"path", "zone", "title", "mtime", "hash"}
        assert isinstance(n["mtime"], int) and n["mtime"] > 1_600_000_000
        assert len(n["hash"]) == 40
    paths = [n["path"] for n in g["nodes"]]
    assert len(paths) == len(set(paths))
    for e in g["edges"]:
        assert len(e) == 2 and e[0] in paths and e[1] in paths and e[0] != e[1]
    assert g["edges"] and "token" in g


def test_order_like_list(env):
    _, http = env
    _, g = http("/graph?limit=300")
    _, lst = http("/list?limit=300")
    assert [n["path"] for n in g["nodes"]] == [d["path"] for d in lst["files"]]
    for n, d in zip(g["nodes"], lst["files"]):
        assert (n["zone"], n["title"]) == (d["zone"], d["title"])


def test_neighbours_like_related(env):
    _, http = env
    _, g = http("/graph?limit=300")
    paths = [n["path"] for n in g["nodes"]]
    inset, expected = set(paths), set()
    for p in paths:
        _, rel = http("/related?path=" + urllib.request.quote(p))
        for t in rel:
            if t["path"] in inset and t["path"] != p:
                expected.add(frozenset((p, t["path"])))
    assert {frozenset(e) for e in g["edges"]} == expected
    assert len(g["edges"]) == len(expected)


def test_score(env):
    _, http = env
    _, g = http("/graph?limit=300&score=1")
    assert all(len(e) == 3 and isinstance(e[2], (int, float)) for e in g["edges"])
    _, o = http("/graph?limit=300")
    assert [e[:2] for e in g["edges"]] == o["edges"]


def test_limit(env):
    _, http = env
    _, g = http("/graph?limit=5")
    _, lst = http("/list?limit=5")
    assert [n["path"] for n in g["nodes"]] == [d["path"] for d in lst["files"]]
    inset = {n["path"] for n in g["nodes"]}
    assert all(e[0] in inset and e[1] in inset for e in g["edges"])


def test_zone(env):
    _, http = env
    _, g = http("/graph?limit=300&zone=midterm")
    assert len(g["nodes"]) == len(TOPICS) and all(n["zone"] == "midterm" for n in g["nodes"])
    _, empty = http("/graph?zone=doesnotexist")
    assert (empty["nodes"], empty["edges"]) == ([], [])


def test_bad_limit(env):
    _, http = env
    with pytest.raises(urllib.error.HTTPError) as e:
        http("/graph?limit=abc")
    assert e.value.code == 400


def test_parallel_requests(env):
    _, http = env
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda _: http("/graph?limit=300"), range(8)))
    assert all(s == 200 for s, _ in res)
    assert len({json.dumps(g["edges"]) for _, g in res}) == 1


def test_parallel_search_requests(env):
    _, http = env
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: http(f"/search?q={TOPICS[i % 10]}&k=2"), range(32)))
    assert all(s == 200 and r["hits"] for s, r in res)


def test_other_endpoints(env):
    _, http = env
    _, st = http("/status")
    assert st["files"] == 21
    _, lst = http("/list?limit=300")
    assert set(lst["files"][0]) == {"path", "zone", "title", "descr", "sections"}
    _, s = http("/search?q=gateway&k=2")
    assert s["hits"]


def test_change_updates_mtime_hash_and_cache(env):
    idx, http = env
    _, before = http("/graph?limit=300")
    _, again = http("/graph?limit=300")
    assert before["token"] == again["token"]
    db = idx.db()
    assert core.graph(db, 300) is core.graph(db, 300)
    rel = "memory/gateway.md"
    old = next(n for n in before["nodes"] if n["path"] == rel)
    time.sleep(1.1)
    write_doc(idx.corpus, rel, "Thema gateway", "geaendert", "Neuer Inhalt zum Gateway und Zebrastreifen.")
    idx.update()
    _, after = http("/graph?limit=300")
    new = next(n for n in after["nodes"] if n["path"] == rel)
    assert old["hash"] != new["hash"] and new["mtime"] > old["mtime"]
    assert before["token"] != after["token"]
    for n in after["nodes"]:
        if n["path"] != rel:
            o = next(x for x in before["nodes"] if x["path"] == n["path"])
            assert (o["hash"], o["mtime"]) == (n["hash"], n["mtime"])
