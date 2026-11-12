"""Secret scrubbing before indexing and embedding (ported from scrub_test.py, plus an index test)."""
import pytest

from memidx import core

CASES = {
    "File `2026-09-14-agent-302-epic-holistic-check-and-more-text.md` stays": False,
    "Link https://host.example:3559/overall-concept-architecture-diagram-final.png": False,
    "Path state/tasks/220/dashboard-src/src/components/status/QuotaPanel.tsx": False,
    "Passwort: geheimes12345": True,
    "password = hunter2hunter2": True,
    "token=abcd1234secretvalue": True,
    "api_key: sk_live_0123456789": True,
    "Key AbCdEf0123456789AbCdEf0123456789AbCdEf0123 end": True,
    "Commit 5b34aedd121e199a97f9d91dec89c98620475cb7 ok": False,
    "The token path /etc/service-board-token is root-only": False,
}


@pytest.mark.parametrize("text,masked", CASES.items())
def test_scrub(text, masked):
    assert (core.MASK in core.scrub(text)) == masked


def test_secret_never_reaches_the_index(idx):
    db = idx.build({"ops.md": "# Ops\n\nThe admin password: Sup3rS3cretValue lives in the vault.\n"})
    body = db.execute("SELECT body FROM chunks").fetchone()[0]
    assert "Sup3rS3cretValue" not in body and core.MASK in body
    assert core.search(db, "Sup3rS3cretValue")["hits"] == []
    assert core.search(db, "vault")["hits"]
