"""Memory entries: own validity rule (marker, JSON field `memory`, include_invalid). Ported from the original
gedaechtnis_test.py; the case list replaces the external case file of the original."""
from datetime import date

import pytest

from memidx import core

TODAY = date(2026, 10, 24)
PREFIX = "memory/entries/"
OUTSIDE = "notes/old-outside.md"

# (case, frontmatter fields, valid, reason, validity text for valid cases)
CASES = [
    ("valid-open", dict(status="bestaetigt", gueltig_ab="2026-01-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01"),
     True, "valid", "until revoked"),
    ("valid-dated", dict(status="bestaetigt", gueltig_ab="2026-01-01", gueltig_bis="2026-12-31", bestaetigt_am="2026-01-01"),
     True, "valid", "until 2026-12-31"),
    ("valid-later-start", dict(status="bestaetigt", gueltig_ab="2026-02-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-15"),
     True, "valid", "from 2026-02-01 until revoked"),
    ("valid-today-ends", dict(status="bestaetigt", gueltig_ab="2026-10-24", gueltig_bis="2026-10-24", bestaetigt_am="2026-10-24"),
     True, "valid", "until 2026-10-24"),
    ("format-status", dict(status="entwurf", gueltig_ab="2026-01-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01"),
     False, "format", None),
    ("format-date", dict(status="bestaetigt", gueltig_ab="2026-13-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01"),
     False, "format", None),
    ("format-order", dict(status="bestaetigt", gueltig_ab="2026-05-01", gueltig_bis="2026-04-01", bestaetigt_am="2026-01-01"),
     False, "format", None),
    ("revoked", dict(status="widerrufen", gueltig_ab="2026-01-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01",
                     widerrufen_am="2026-03-01"), False, "revoked", None),
    ("superseded", dict(status="ersetzt", gueltig_ab="2026-01-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01"),
     False, "superseded", None),
    ("superseded-by", dict(status="bestaetigt", gueltig_ab="2026-01-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-01-01",
                           ersetzt_durch="memory/entries/valid-open.md"), False, "superseded", None),
    ("not-yet", dict(status="bestaetigt", gueltig_ab="2026-11-01", gueltig_bis="bis_widerruf", bestaetigt_am="2026-10-01"),
     False, "not_yet_valid", None),
    ("expired", dict(status="bestaetigt", gueltig_ab="2026-01-01", gueltig_bis="2026-10-23", bestaetigt_am="2026-01-01"),
     False, "expired", None),
    # English keys and values
    ("en-valid", dict(status="confirmed", valid_from="2026-01-01", valid_until="until_revoked", confirmed_on="2026-01-01",
                      kind="preference"), True, "valid", "until revoked"),
    ("en-revoked", dict(status="revoked", valid_from="2026-01-01", valid_until="until_revoked", confirmed_on="2026-01-01"),
     False, "revoked", None),
    ("en-superseded-by", dict(status="confirmed", valid_from="2026-01-01", valid_until="2027-01-01", confirmed_on="2026-01-01",
                              superseded_by="x.md"), False, "superseded", None),
]


def entry(case, fields):
    lines = ["---", f"name: fact-{case}", f"description: Test content {case} keyword"]
    if "kind" not in fields:
        lines.append("art: fakt")
    lines += [f"{k}: {v}" for k, v in fields.items()]
    lines += ["---", "", f"Test content for case {case} with the search word keyword.", ""]
    return "\n".join(lines)


@pytest.fixture
def db(idx):
    files = {f"{PREFIX}{c}.md": entry(c, f) for c, f, *_ in CASES}
    files[OUTSIDE] = ("---\nname: old-outside\ndescription: Test content outside keyword\ngueltig_bis: 2020-01-01\n---\n\n"
                      "Test content outside the prefix with the search word keyword.\n")
    return idx.build(files, memory_entries_prefix=PREFIX)


def hits(db, include_invalid=False):
    res = core._search(db, "keyword", 50, 20000, None, 5, 1400, False, True, include_invalid, TODAY)
    return {t["path"]: t for t in res["hits"]}


def test_valid_cases_carry_marker_and_json_field(db):
    t = hits(db)
    for case, fields, valid, reason, text in CASES:
        if not valid:
            continue
        p = f"{PREFIX}{case}.md"
        assert p in t, case
        g = t[p]["memory"]
        assert g["valid"] and g["reason"] == "valid"
        kind = "Preference" if fields.get("kind") == "preference" else "Fact"
        confirmed = fields.get("bestaetigt_am") or fields.get("confirmed_on")
        mark = f"[MEMORY {kind}, valid {text}, confirmed {confirmed}]"
        single = {"hints": [], "hits": [t[p]]}
        assert mark in core.render_text(single)
        assert mark in core.render_list(single)


def test_invalid_cases_missing_without_include_invalid(db):
    t = hits(db, False)
    for case, _, valid, *_ in CASES:
        if not valid:
            assert f"{PREFIX}{case}.md" not in t, case
    res = {"hints": [], "hits": list(t.values())}
    assert "invalid" not in core.render_text(res)
    assert "invalid" not in core.render_list(res)


def test_invalid_cases_appear_with_reason(db):
    t = hits(db, True)
    for case, _, valid, reason, _ in CASES:
        if valid:
            continue
        p = f"{PREFIX}{case}.md"
        assert p in t, case
        assert t[p]["memory"]["valid"] is False
        assert t[p]["memory"]["reason"] == reason, case
        assert f"[MEMORY invalid: {reason}]" in core.render_text({"hints": [], "hits": [t[p]]})


def test_related_drops_invalid_unless_include_invalid(db):
    anchor = f"{PREFIX}valid-open.md"
    bad = f"{PREFIX}expired.md"
    without = {t["path"] for t in core.related(db, anchor, k=50, private_ok=True, today=TODAY)}
    with_all = {t["path"] for t in core.related(db, anchor, k=50, private_ok=True, include_invalid=True, today=TODAY)}
    assert bad not in without
    assert bad in with_all


def test_document_outside_prefix_stays_visible_and_stale(db):
    e = hits(db)[OUTSIDE]
    assert e.get("stale") and "memory" not in e
    assert "[STALE]" in core.render_text({"hints": [], "hits": [e]})


def test_rule_off_by_default(idx):
    db = idx.build({f"{PREFIX}x.md": entry("x", CASES[-2][1])})
    t = hits(db)
    assert "memory" not in t[f"{PREFIX}x.md"]
