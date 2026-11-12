"""Validity and supersession: stale documents are down-weighted and marked (ported from gueltigkeit_test.py)."""
import time

import pytest

from memidx import core

DOCS = {
    "notes/new.md": "---\nname: new\ndescription: current description of the zebra-crossing service\n---\n# New\nThe zebra-crossing service runs on host-b.\n",
    "notes/old.md": "---\nname: old\ndescription: old description of the zebra-crossing service\nersetzt_durch: notes/new.md\n---\n# Old\nThe zebra-crossing service runs on host-a.\n",
    "notes/expired.md": "---\nname: expired\ndescription: temporary zebra-crossing service\ngueltig_bis: 2020-01-01\n---\n# Temporary\nThe zebra-crossing service was temporary.\n",
    "notes/valid.md": "---\nname: valid\ndescription: temporary zebra-crossing service valid\ngueltig_bis: 2999-12-31\n---\n# Valid\nThe zebra-crossing service is temporary but valid.\n",
    "notes/en-old.md": "---\nname: en-old\ndescription: zebra-crossing service english keys\nsuperseded_by: notes/new.md\n---\n# EnOld\nzebra-crossing service.\n",
    "notes/en-expired.md": "---\nname: en-expired\ndescription: zebra-crossing service english expiry\nvalid_until: 2020-01-01\n---\n# EnExp\nzebra-crossing service.\n",
}


@pytest.fixture
def db(idx):
    return idx.build(DOCS)


def search(db):
    return core._search(db, "zebra-crossing service", 10, 4000, None, 2, 1400, False, True)


def by_path(db):
    return {t["path"]: t for t in search(db)["hits"]}


def test_superseded_by_is_marked_and_shown(db):
    t = by_path(db)["notes/old.md"]
    assert t.get("stale") and t["superseded_by"] == "notes/new.md"


def test_english_keys(db):
    t = by_path(db)
    assert t["notes/en-old.md"]["superseded_by"] == "notes/new.md"
    assert t["notes/en-expired.md"].get("stale")


def test_expired_valid_until_is_stale_future_is_not(db):
    t = by_path(db)
    assert t["notes/expired.md"].get("stale")
    assert not t["notes/valid.md"].get("stale")
    assert not t["notes/new.md"].get("stale")


def test_stale_ranks_behind_current(db):
    paths = [t["path"] for t in search(db)["hits"]]
    assert paths.index("notes/new.md") < paths.index("notes/old.md")


def test_output_carries_marker(db):
    res = search(db)
    assert "[STALE, superseded by notes/new.md]" in core.render_text(res)
    assert "[STALE, superseded by notes/new.md]" in core.render_list(res)


def test_recency_off_by_default(idx):
    idx.configure()
    assert core.recency_factor({"mtime": 1.0}) == 1.0
    idx.configure(ranking={"recency": True})
    fresh = core.recency_factor({"mtime": time.time()})
    old = core.recency_factor({"mtime": time.time() - 3650 * 86400})
    assert fresh > 1.1 and old < 1.01
