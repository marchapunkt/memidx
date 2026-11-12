"""Shared fixtures: every test gets its own temporary corpus, index and configuration."""
import os
from dataclasses import replace

import pytest

from memidx import core, embed
from memidx.config import Config


def write(corpus, rel, text):
    p = os.path.join(corpus, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_doc(corpus, rel, title, descr, text):
    write(corpus, rel, f"---\nname: {title}\ndescription: {descr}\n---\n# {title}\n\n{text}\n")


def quiet(*a):
    pass


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for k in list(os.environ):
        if k.startswith("MEMIDX_"):
            monkeypatch.delenv(k, raising=False)
    embed.set_embedder(None)
    core.configure(Config())
    yield
    embed.set_embedder(None)
    core.configure(Config())


class Index:
    """A temporary corpus plus index; `build()` writes files, configures and indexes."""

    def __init__(self, root):
        self.root = str(root)
        self.corpus = os.path.join(self.root, "corpus")
        self.dbpath = os.path.join(self.root, "m.db")
        os.makedirs(self.corpus, exist_ok=True)
        self.cfg = None
        self._conns = []

    def configure(self, **kw):
        base = Config(corpus=self.corpus, db=self.dbpath, model_cache=os.path.join(self.root, "models"))
        if "ranking" in kw and isinstance(kw["ranking"], dict):
            kw["ranking"] = replace(base.ranking, **kw["ranking"])
        self.cfg = core.configure(replace(base, **kw))
        return self.cfg

    def build(self, files=None, docs=None, embed=False, **kw):
        for rel, text in (files or {}).items():
            write(self.corpus, rel, text)
        for rel, (title, descr, text) in (docs or {}).items():
            write_doc(self.corpus, rel, title, descr, text)
        if kw or self.cfg is None:
            self.configure(**kw)
        self.update(embed=embed)
        return self.db()

    def update(self, embed=False):
        return core.update(self.corpus, self.dbpath, embed=embed, log=quiet)

    def db(self):
        c = core.open_db(self.dbpath)
        self._conns.append(c)
        return c

    def close(self):
        for c in self._conns:
            c.close()


@pytest.fixture
def idx(tmp_path):
    i = Index(tmp_path)
    yield i
    i.close()
