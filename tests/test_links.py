"""Wikilinks as edges and the link bonus in ranking (ported from links_test.py)."""
import pytest

from memidx import core

# cycle alpha -> beta -> gamma -> alpha; delta is a hub (links alpha, beta, gamma); epsilon has only broken
# links and a self link; zeta links by frontmatter name and with alias/section.
NOTES = {
    "memory/alpha.md": ("Alpha", "Fuchs Rezept Zutaten. Weiter mit [[beta]]."),
    "memory/beta.md": ("Beta", "Fuchs Rezept Salz. Weiter mit [[gamma.md]]."),
    "memory/gamma.md": ("Gamma", "Fuchs Rezept Pfeffer. Zurueck zu [[Alpha]]."),
    "memory/delta.md": ("Delta", "Fuchs Rezept Uebersicht: [[alpha]] [[beta|der zweite]] [[gamma#Teil]]."),
    "memory/epsilon.md": ("Epsilon", "Fuchs Rezept Notizen: [[gibt-es-nicht]] [[epsilon]] [[ ]] [[]] und `[[im-code]]`."),
    "memory/zeta.md": ("Zeta", "Quokka Zebrastreifen ohne Bezug. Siehe [[memory/alpha]] und [[Delta]]."),
}


@pytest.fixture
def db(idx):
    return idx.build(docs={rel: (n, n, t) for rel, (n, t) in NOTES.items()})


def edges(db):
    return {(a, b) for a, b in db.execute("SELECT src, dst FROM links")}


def test_parser():
    assert core.wikilinks("[[a]] [[b|x]] [[c#h]] [[a]] `[[d]]` ```\n[[e]]\n```") == ["a", "b", "c"]


def test_edges_resolved(db):
    k = edges(db)
    for a, b in [("alpha", "beta"), ("beta", "gamma"), ("gamma", "alpha"), ("delta", "alpha"), ("delta", "beta"),
                 ("delta", "gamma"), ("zeta", "alpha"), ("zeta", "delta")]:
        assert (f"memory/{a}.md", f"memory/{b}.md") in k
    assert len(k) == 8
    assert not [1 for a, b in k if a == b]
    assert not [1 for a, b in k if a.endswith("epsilon.md")]


def test_reindex_keeps_edges(idx, db):
    idx.update()
    assert len(edges(db)) == 8


def test_related_real_edges_first(db):
    r = core.related(db, "memory/alpha.md", k=5)
    paths = [t["path"] for t in r]
    assert {"memory/beta.md", "memory/gamma.md", "memory/delta.md", "memory/zeta.md"} <= set(paths)
    assert all(t.get("via") == "link" for t in r[:4])
    assert "memory/alpha.md" not in paths
    assert len(paths) == len(set(paths))


def test_related_isolated_and_broken(db):
    r = core.related(db, "memory/epsilon.md", k=3)
    assert r and not [t for t in r if t.get("via") == "link"]


def test_graph_uses_edges(db):
    g = core.graph(db, 50)
    pairs = {frozenset(e[:2]) for e in g["edges"]}
    assert frozenset(("memory/zeta.md", "memory/alpha.md")) in pairs


def test_bonus_bounded_and_switchable(idx, db):
    q = dict(k=6, budget_tokens=5000, per_file=1)
    idx.configure(ranking={"links": False})
    off = {t["path"]: t["score"] for t in core.search(db, "fuchs rezept", **q)["hits"]}
    idx.configure(ranking={"links": True})
    on = core.search(db, "fuchs rezept", **q)["hits"]
    for t in on:
        assert t["score"] >= off[t["path"]] - 0.011
        assert t["score"] <= off[t["path"]] * 1.10 + 0.011     # at most +10 %
    assert any(t["score"] > off[t["path"]] + 0.011 for t in on)
    idx.configure(ranking={"links": True, "links_max": 0.0})
    assert {t["path"]: t["score"] for t in core.search(db, "fuchs rezept", **q)["hits"]} == off


def test_ranking_without_links_unchanged(db):
    assert core.search(db, "quokka zebrastreifen", k=3)["hits"][0]["path"] == "memory/zeta.md"
