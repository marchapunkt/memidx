"""HTTP smoke test: the real `serve()` entry point on an ephemeral port."""
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from memidx import core, server


@pytest.fixture
def served(idx):
    idx.configure(vectors=False)
    from conftest import write_doc
    write_doc(idx.corpus, "notes/a.md", "Alpha", "first", "The quokka lives on Rottnest island.")
    write_doc(idx.corpus, "notes/b.md", "Beta", "second", "Quokka photos and [[Alpha]].")
    box = {}
    ready = threading.Event()

    def on_ready(httpd):
        box["httpd"] = httpd
        ready.set()
    t = threading.Thread(target=server.serve, kwargs=dict(cfg=idx.cfg, host="127.0.0.1", port=0, interval=0, auto=True,
                                                          ready=on_ready), daemon=True)
    t.start()
    assert ready.wait(30)
    port = box["httpd"].server_address[1]

    def get(path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
                return r.status, r.headers.get("Content-Type"), r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read().decode("utf-8")
    yield get
    box["httpd"].shutdown()
    t.join(10)


def test_endpoints(served):
    s, ct, body = served("/status")
    assert s == 200 and json.loads(body)["files"] == 2
    s, ct, body = served("/search?q=quokka&k=5")
    res = json.loads(body)
    assert s == 200 and ct.startswith("application/json") and {t["path"] for t in res["hits"]} == {"notes/a.md", "notes/b.md"}
    assert set(res["hits"][0]) >= {"path", "section", "lines", "zone", "score", "text"}
    s, ct, body = served("/search?q=quokka&mode=text")
    assert s == 200 and body.startswith("== notes/")
    s, ct, body = served("/search?q=quokka&mode=list")
    assert body.startswith("1. notes/")
    s, _, body = served("/doc?path=notes/a.md&from=1&to=0")
    assert s == 200 and "Rottnest" in body
    assert served("/doc?path=nope.md")[0] == 404
    s, _, body = served("/related?path=notes/b.md")
    assert json.loads(body)[0] == {**json.loads(body)[0], "path": "notes/a.md", "via": "link"}
    s, _, body = served("/list")
    assert [f["path"] for f in json.loads(body)["files"]] == ["notes/a.md", "notes/b.md"]
    s, _, body = served("/graph?limit=10")
    assert json.loads(body)["n"] == 2
    assert served("/nothing")[0] == 404
    assert served("/search?q=x&k=abc")[0] == 400


def test_ui(served):
    s, ct, body = served("/?q=" + urllib.parse.quote("quokka <script>"))
    assert s == 200 and ct.startswith("text/html")
    assert "Rottnest" in body and "<script>" not in body and "Search" in body


def test_default_bind_address():
    from memidx.config import Config
    assert Config().host == "127.0.0.1" and Config().mcp_host == "127.0.0.1"
