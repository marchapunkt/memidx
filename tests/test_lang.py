"""Normalization: German rules unchanged, English stemmer, combined rule, rebuild on language change."""
import pytest

from memidx import core, lang
from memidx.lang import Normalizer, stem_de, stem_en

GERMAN = ["Zeitungen", "Häuser", "Freiheiten", "Gesellschaft", "Möglichkeit", "Straße", "Dienste", "Rechnern",
          "Wissens", "Sicherung", "laufen", "Gateways", "Übersicht", "Kindern", "großes"]


def test_german_stems_match_original_rules():
    expected = {"zeitungen": "zeit", "hauser": "haus", "freiheiten": "frei", "gesellschaft": "gesell", "moglichkeit": "moglich",
                "strasse": "strass", "dienste": "dienst", "rechnern": "rechn", "wissens": "wissen", "sicherung": "sicher",
                "laufen": "lauf", "gateways": "gateway", "ubersicht": "ubersicht", "kindern": "kind", "grosses": "gross"}
    de = Normalizer(["de"])
    assert {lang.fold(w): de.stem(lang.fold(w)) for w in GERMAN} == expected


def test_german_stopwords_kept():
    assert Normalizer(["de"]).terms("Der Dienst ist nicht fuer die Sicherung") == ["dienst", "sicher"]


@pytest.mark.parametrize("word,stem", [("queries", "queri"), ("query", "queri"), ("indexes", "index"), ("indexed", "index"),
                                       ("indexing", "index"), ("tests", "test"), ("running", "run"), ("stopped", "stop"),
                                       ("falling", "fall"), ("class", "class"), ("status", "status"), ("analysis", "analysis"),
                                       ("boxes", "box"), ("used", "used"), ("bus", "bus"), ("2026", "2026"), ("cat", "cat")])
def test_english_stems(word, stem):
    assert stem_en(word) == stem


def test_english_terms():
    assert Normalizer(["en"]).terms("The indexes were rebuilt by running queries") == ["index", "rebuilt", "run", "queri"]


def test_combined_rule_german_first():
    both = Normalizer(["de", "en"])
    for w in GERMAN:
        f = lang.fold(w)
        if stem_de(f) != f:
            assert both.stem(f) == stem_de(f), w       # German stems are never changed by the English rules
    assert both.stem("indexing") == "index"            # German leaves it alone, English strips -ing
    assert both.stem("queries") == both.stem("query") == "queri"
    assert both.stop == lang.STOP_DE | lang.STOP_EN


def test_order_is_fixed():
    assert Normalizer(["en", "de"]).languages == ("de", "en")
    assert Normalizer(["en", "de"]).signature == Normalizer(["de", "en"]).signature


def test_unsupported_language():
    with pytest.raises(ValueError):
        Normalizer(["fr"])
    with pytest.raises(ValueError):
        Normalizer([])


def test_language_change_rebuilds_fulltext(idx):
    db = idx.build({"a.md": "# Jobs\n\nThe worker runs nightly.\n", "b.md": "# Other\n\nnothing here\n"}, languages=("de",))
    assert core.search(db, "running")["hits"] == []                  # German rules: "running" != "runs"
    sig_de = core.status(db)["normalization"]
    idx.configure(languages=("en",))
    r = idx.update()
    assert r["changed"] == 0                                          # no re-chunking, only the full-text table
    assert core.status(db)["normalization"] != sig_de
    assert [t["path"] for t in core.search(db, "running")["hits"]] == ["a.md"]
    # deletes in the contentless FTS table still match after the rebuild
    with open(f"{idx.corpus}/a.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("# Jobs\n\nThe worker sleeps.\n")
    idx.update()
    assert core.search(db, "running")["hits"] == []
    assert [t["path"] for t in core.search(db, "sleeping")["hits"]] == ["a.md"]
    assert db.execute("SELECT count(*) FROM fts WHERE fts MATCH 'run'").fetchone()[0] == 0


def test_same_languages_no_rebuild(idx, capsys):
    idx.build({"a.md": "# A\n\nword\n"})
    msgs = []
    core.update(idx.corpus, idx.dbpath, log=msgs.append)
    assert not [m for m in msgs if "Normalization changed" in m]
