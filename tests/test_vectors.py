"""Vector path with a deterministic fake embedder (ported from synonym_test.py) and FTS-only mode."""
import os
import sys

import pytest

from memidx import core, embed
from memidx.config import ZoneRule
from memidx.lang import fold

np = pytest.importorskip("numpy")

# concept axes of the fake: words of one concept point to the same axis, even without a shared word form
CONCEPTS = [("aehnlichkeitssuche", "semantisch", "vektor", "wissensabruf"), ("backup", "sicherung"), ("tunnel", "fernzugang")]


def _vec(text):
    t = fold(text)
    v = [float(sum(t.count(fold(w)) for w in c)) for c in CONCEPTS] + [0.05]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


class FakeEmbedder:
    name = "fake-concepts-v1"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


def create_fake(cfg=None):          # factory for the `embedder = "module:factory"` hook
    return FakeEmbedder()


PRESET = "embeddinggemma-300m"
TARGET = "memory/decisions/topic-search.md"
ZONES = (ZoneRule("scratch", 0.6, basename="worklog*"), ZoneRule("longterm", 1.0, prefix="memory/"))


def write(corpus, rel, title, text, descr=None):
    p = os.path.join(corpus, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        head = f"---\nname: {title}\ndescription: {descr}\n---\n" if descr else ""
        f.write(f"{head}# {title}\n\n{text}\n")


@pytest.fixture
def vidx(idx):
    fake = FakeEmbedder()
    embed.set_embedder(fake)
    c = idx.corpus
    # target: sections without a concept word of their own; only the description names the topic
    write(c, TARGET, "Suche", "## Aufbau\n\nDienst im Container, Port 8800.\n\n## Betrieb\n\nNeustart durch systemd.",
          descr="semantisch, Vektor")
    write(c, "memory/decisions/topic-borg.md", "Borg", "## Sicherung\n\nbackup sicherung backup nachts.")
    write(c, "memory/decisions/topic-tunnel.md", "Tunnel", "## Zugang\n\ntunnel fernzugang tunnel.")
    for i in range(5):
        write(c, f"memory/note{i}.md", f"Note{i}", "## Teil\n\nvektor " + "backup tunnel " * (i + 1))
    write(c, "projects/p/worklog.md", "Stand", "## Teil\n\nsemantisch vektor semantisch.")
    db = idx.build(embed=True, zones=ZONES, vectors=True)
    idx.fake = fake
    return idx, db


def test_document_vectors_only_for_vector_zones(vidx):
    _, db = vidx
    have = {p for (p,) in db.execute("SELECT d.path FROM dvec v JOIN docs d ON d.id=v.doc")}
    assert TARGET in have and "projects/p/worklog.md" not in have        # weight 0.6 < 0.7: no vector
    assert db.execute("SELECT count(*) FROM dvec").fetchone()[0] == \
        db.execute("SELECT count(*) FROM docs WHERE prior >= 0.7").fetchone()[0]
    assert db.execute("SELECT count(*) FROM vec v JOIN chunks c ON c.id=v.id JOIN docs d ON d.id=c.doc "
                      "WHERE d.path='projects/p/worklog.md'").fetchone()[0] == 0


def test_unchanged_documents_not_embedded_again(vidx):
    idx, db = vidx
    calls = idx.fake.calls
    assert core.embed_missing_docs(db, log=lambda m: None) == 0
    r = idx.update(embed=True)
    assert r["new_vectors"] == 0 and idx.fake.calls == calls


def test_changed_section_reembedded_only(vidx):
    idx, db = vidx
    calls = idx.fake.calls
    write(idx.corpus, "memory/note0.md", "Note0", "## Teil\n\nvektor backup tunnel \n\n## Neu\n\nfernzugang")
    r = idx.update(embed=True)
    assert r["changed"] == 1 and r["new_vectors"] == 1          # the unchanged section reuses its vector
    assert idx.fake.calls - calls == 2                          # one section + one document vector


def test_stale_document_vector_removed_without_embed(vidx):
    idx, db = vidx
    p = os.path.join(idx.corpus, "memory/note4.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Note4\n\n## Teil\n\n" + "vektor backup " * core.DOC_VEC_CHARS + "\n")
    idx.update(embed=True)
    did = db.execute("SELECT id FROM docs WHERE path='memory/note4.md'").fetchone()[0]
    count = lambda: db.execute("SELECT count(*) FROM dvec WHERE doc=?", (did,)).fetchone()[0]
    assert count() == 1
    with open(p, "a", encoding="utf-8", newline="\n") as f:        # change after the embedded characters: kept
        f.write("\n## Later\n\nnew paragraph\n")
    idx.update(embed=False)
    assert count() == 1
    with open(p, "w", encoding="utf-8", newline="\n") as f:        # change in the embedded start: dropped
        f.write("# Note4\n\n## Teil\n\ncompletely different start\n")
    idx.update(embed=False)
    assert count() == 0
    os.remove(p)
    idx.update(embed=False)
    assert db.execute("SELECT count(*) FROM dvec WHERE doc NOT IN (SELECT id FROM docs)").fetchone()[0] == 0


def test_cache_detects_replaced_vectors(vidx):
    _, db = vidx
    ids, mat = core.doc_vec_matrix(db)
    new = np.asarray([0.5] * mat.shape[1], dtype="float32")
    db.execute("UPDATE dvec SET h='other', v=? WHERE doc=?", (new.tobytes(), int(ids[0])))
    db.commit()
    ids2, mat2 = core.doc_vec_matrix(db)
    assert len(ids2) == len(ids) and np.allclose(mat2[list(ids2).index(ids[0])], new)


def test_index_without_dvec_table():
    import sqlite3
    db = sqlite3.connect(":memory:")
    assert core.doc_vec_matrix(db) == (None, None)
    db.close()


def _rank(idx, db, question, w_doc):
    idx.configure(zones=ZONES, vectors=True, ranking={"rrf_doc": w_doc})
    paths = [t["path"] for t in core.search(db, question, k=10, budget_tokens=100000, per_file=1)["hits"]]
    return paths.index(TARGET) + 1 if TARGET in paths else 99


def test_paraphrase_ranked_better_by_document_vector(vidx):
    idx, db = vidx
    q = "Wissensabruf per Ähnlichkeitssuche"   # no word of the question is in the target
    without, with_doc = _rank(idx, db, q, 0.0), _rank(idx, db, q, 0.3)
    assert without > 1 and with_doc < without


def test_hybrid_finds_paraphrase_that_fts_misses(vidx):
    idx, db = vidx
    idx.configure(zones=ZONES, vectors=True)
    q = "fernzugang"
    fts = [t["path"] for t in core.search(db, q, k=10, per_file=1, hybrid=False)["hits"]]
    hyb = [t["path"] for t in core.search(db, q, k=10, per_file=1)["hits"]]
    assert "memory/note0.md" not in fts and "memory/note0.md" in hyb   # note0 says "tunnel", not "fernzugang"


def test_embedder_change_discards_vectors(vidx):
    idx, db = vidx

    class Other(FakeEmbedder):
        name = "other-model"
    other = Other()
    embed.set_embedder(other)
    idx.update(embed=True)
    assert db.execute("SELECT v FROM meta WHERE k='embedder'").fetchone()[0] == "other-model"
    assert other.calls == db.execute("SELECT count(*) FROM vec").fetchone()[0] + db.execute("SELECT count(*) FROM dvec").fetchone()[0]


def test_embedder_hook_from_config(idx):
    embed.set_embedder(None)
    db = idx.build({"memory/a.md": "# A\n\ntunnel"}, embed=True, embedder="test_vectors:create_fake", vectors=True)
    assert db.execute("SELECT count(*) FROM vec").fetchone()[0] == 1


def test_fts_only_without_vector_dependencies(idx, monkeypatch, capsys):
    for mod in ("numpy", "onnxruntime", "tokenizers"):
        monkeypatch.setitem(sys.modules, mod, None)              # import now raises ImportError
    embed._NOTICE_SHOWN.clear()
    db = idx.build({"memory/a.md": "# A\n\nTunnel und Fernzugang.\n"}, embed=True, vectors=True, model=PRESET)
    assert db.execute("SELECT count(*) FROM vec").fetchone()[0] == 0
    res = core.search(db, "fernzugang")
    assert [t["path"] for t in res["hits"]] == ["memory/a.md"]
    err = capsys.readouterr().err
    assert err.count("full-text search only") == 1 and len(err.strip().splitlines()) == 1


def test_missing_model_files_fall_back(idx, capsys):
    embed._NOTICE_SHOWN.clear()
    if not embed.deps_available():
        pytest.skip("vector extra not installed")
    db = idx.build({"memory/a.md": "# A\n\nword\n"}, embed=True, vectors=True, model=PRESET)
    assert core.search(db, "word")["hits"]
    assert "memidx model download" in capsys.readouterr().err


def test_vectors_disabled_by_config(idx, capsys):
    fake = FakeEmbedder()
    embed.set_embedder(fake)
    db = idx.build({"memory/a.md": "# A\n\nword\n"}, embed=True, vectors=False)
    assert fake.calls == 0 and db.execute("SELECT count(*) FROM vec").fetchone()[0] == 0
