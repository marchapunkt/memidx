"""Hints from [[hints]] and [[renames]] (rewritten from hints_test.py, which read an external name table)."""
import pytest

from memidx import core
from memidx.config import Config

RENAMES = (
    {"old": "ravenwatch", "new": "reviewer", "since": "2025-03-01"},
    {"old": "büro", "new": "requirements", "since": "2025-03-01", "variants": ["buero"]},
    {"old": ".corp", "new": ".example", "since": "2025-03-01"},
    {"old": "gone-host", "new": "", "since": "2025-04-01"},          # no new name: no rename hint
)
HINTS = (
    (r"\bwikiold\b", "WikiOld was shut down; notes live in notes/."),
    (r"oldhost", "oldhost no longer resolves; the machine is called workhost."),
)


@pytest.fixture(autouse=True)
def cfg():
    core.configure(Config(renames=RENAMES, hints=HINTS))


def text(r):
    return f'{r["old"]} is called {r["new"]} since {r["since"]}.'


def test_rename_gives_exactly_one_hint():
    h = core.hints("ravenwatch")
    assert h == [text(RENAMES[0])]


def test_every_rename_with_new_name_hints():
    for r in RENAMES:
        if r["new"]:
            assert text(r) in core.hints(r["old"]), r


def test_rename_without_new_name_gives_no_hint():
    assert core.hints("gone-host") == []


def test_variants_give_the_same_hint():
    assert core.hints("buero") == [text(RENAMES[1])]
    assert core.hints("Büro") == [text(RENAMES[1])]


def test_name_starting_with_non_word_char():
    assert core.hints(".corp") == [text(RENAMES[2])]
    assert core.hints("the .corp domain") == [text(RENAMES[2])]


def test_word_part_does_not_trigger():
    assert core.hints("ravenwatchtower") == []


def test_configured_hints():
    assert core.hints("Where is WikiOld?") == [HINTS[0][1]]
    assert core.hints("oldhost") == [HINTS[1][1]]


def test_hint_is_first_line_of_render_text():
    first = core.render_text({"hints": core.hints("ravenwatch"), "hits": []}).splitlines()[0]
    assert first.startswith("[Hint] ravenwatch is called reviewer")


def test_no_hints_by_default():
    core.configure(Config())
    assert core.hints("ravenwatch wikiold") == []


def test_hints_appear_in_search(idx):
    db = idx.build({"a.md": "# A\n\nravenwatch checks things.\n"}, renames=RENAMES)
    res = core.search(db, "ravenwatch")
    assert res["hints"] == [text(RENAMES[0])] and res["hits"]
