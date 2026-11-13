"""Configuration loading, precedence, environment overrides and neutral defaults."""
import json
import os
from datetime import date

import pytest

from memidx import core
from memidx.config import EXAMPLE, EXAMPLE_CORPUS, Config, ZoneRule, default_db_path, is_inside, load_config, today_in


def toml(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_neutral_defaults(tmp_path):
    cfg = load_config(corpus=str(tmp_path), env={})
    assert cfg.source == ""
    assert cfg.zone_of("anything/deep/file.md") == ("default", 1.0)
    assert cfg.zone_of("arbeitsstand.md") == ("default", 1.0)          # no special file-name rule
    assert cfg.all_hints() == [] and cfg.private_prefixes == () and cfg.exclude == ()
    assert cfg.memory_entries_prefix == "" and cfg.log_path == ""
    assert cfg.languages == ("de", "en")
    assert cfg.host == "127.0.0.1" and cfg.mcp_host == "127.0.0.1"
    assert cfg.db == default_db_path(tmp_path, {}) and not is_inside(cfg.db, tmp_path)   # outside the corpus
    assert cfg.vectors is False and cfg.model == "" and cfg.git_safe_directory is False
    core.configure(cfg)
    assert not core.sees_private("own:me") or cfg.private_prefixes == ()
    assert not core.is_private("private/x.md")


def test_example_config_parses_to_defaults(tmp_path):
    p = toml(tmp_path / "memidx.toml", EXAMPLE_CORPUS)
    cfg = load_config(corpus=str(tmp_path), env={})
    assert cfg.source == str(p) and cfg.notices == ()                    # corpus example: only allowed keys
    full = toml(tmp_path / "user" / "full.toml", EXAMPLE)
    cfg2 = load_config(str(full), corpus=str(tmp_path / "other"), env={})
    d = Config()
    for f in ("vectors", "model", "git_safe_directory", "private_prefixes", "own_client_prefix", "private_mode",
              "log_path", "host", "port", "mcp_host", "mcp_port", "mcp_client", "interval", "ranking", "default_zone",
              "vector_min_weight", "memory_entries_prefix"):
        assert getattr(cfg2, f) == getattr(d, f), f
    for f in ("languages", "timezone", "zones", "default_zone", "vector_min_weight", "exclude", "hints", "renames",
              "private_prefixes", "own_client_prefix", "private_mode", "memory_entries_prefix", "ranking", "log_path",
              "vectors", "host", "port", "mcp_host", "mcp_port", "mcp_client", "interval"):
        assert getattr(cfg, f) == getattr(d, f), f


def test_precedence_explicit_env_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    toml(corpus / "memidx.toml", 'timezone = "from-corpus"')
    env_file = toml(tmp_path / "env.toml", 'timezone = "from-env"')
    explicit = toml(tmp_path / "explicit.toml", 'timezone = "from-explicit"')
    assert load_config(corpus=str(corpus), env={}).timezone == "from-corpus"
    assert load_config(corpus=str(corpus), env={"MEMIDX_CONFIG": str(env_file)}).timezone == "from-env"
    assert load_config(str(explicit), corpus=str(corpus), env={"MEMIDX_CONFIG": str(env_file)}).timezone == "from-explicit"


def test_environment_overrides_file(tmp_path):
    toml(tmp_path / "memidx.toml", """
exclude = ["a/"]
vectors = true
private_prefixes = ["p/"]
[ranking]
rrf_doc = 0.5
""")
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps([["docs/", "docs", 0.9]]), encoding="utf-8")
    env = {"MEMIDX_EXCLUDE": "b,c", "MEMIDX_VECTORS": "0", "MEMIDX_PRIVATE_PREFIXES": "", "MEMIDX_RRF_DOC": "0",
           "MEMIDX_LINKS": "0", "MEMIDX_ZONES": str(zones), "MEMIDX_LOG": "/tmp/q.jsonl", "MEMIDX_LANGUAGES": "en",
           "MEMIDX_DB": "/data/x.db", "MEMIDX_GEDAECHTNIS_PREFIX": "mem/"}
    cfg = load_config(corpus=str(tmp_path), env=env)
    assert cfg.exclude == ("b", "c") and cfg.vectors is False and cfg.private_prefixes == ()
    assert cfg.ranking.rrf_doc == 0.0 and cfg.ranking.links is False
    assert cfg.zones == (ZoneRule("docs", 0.9, prefix="docs/"),)
    assert cfg.log_path == "/tmp/q.jsonl" and cfg.languages == ("en",) and cfg.db == "/data/x.db"
    assert cfg.memory_entries_prefix == "mem/"


def test_cli_db_flag_wins(tmp_path):
    cfg = load_config(corpus=str(tmp_path), db=str(tmp_path / "flag.db"), env={"MEMIDX_DB": "/env.db"})
    assert cfg.db == str(tmp_path / "flag.db")


def test_relative_paths_are_relative_to_config_file(tmp_path):
    toml(tmp_path / "cfg" / "memidx.toml", 'db = "idx/i.db"\nlog_path = "q.jsonl"')
    cfg = load_config(str(tmp_path / "cfg" / "memidx.toml"), corpus=str(tmp_path), env={})
    assert cfg.db == str((tmp_path / "cfg" / "idx" / "i.db").resolve())
    assert cfg.log_path == str((tmp_path / "cfg" / "q.jsonl").resolve())


@pytest.mark.parametrize("text", ['unknown_key = 1', '[ranking]\nrrf_bm = 1.0', 'languages = ["fr"]',
                                  '[[zones]]\nname = "x"', '[[zones]]\nname = "x"\nprefix = "a/"\nbasename = "b*"',
                                  '[server]\nprot = 1'])
def test_invalid_config_raises(tmp_path, text):
    p = toml(tmp_path / "trusted.toml", text)                             # trusted file: strict
    with pytest.raises(ValueError):
        load_config(str(p), corpus=str(tmp_path), env={})


def test_zone_rules(tmp_path):
    toml(tmp_path / "memidx.toml", """
[default_zone]
name = "other"
weight = 0.6
[[zones]]
prefix = "archive/old/"
name = "history"
weight = 0.3
[[zones]]
prefix = "archive/"
name = "archive"
weight = 0.5
[[zones]]
prefix = "README.md"
name = "rules"
weight = 1.0
[[zones]]
basename = "worklog*"
name = "scratch"
weight = 0.6
""")
    cfg = load_config(corpus=str(tmp_path), env={})
    assert cfg.zone_of("archive/old/a.md") == ("history", 0.3)
    assert cfg.zone_of("archive/b.md") == ("archive", 0.5)
    assert cfg.zone_of("README.md") == ("rules", 1.0)
    assert cfg.zone_of("docs/README.md") == ("other", 0.6)
    assert cfg.zone_of("archive/old/worklog-12.md") == ("scratch", 0.6)      # basename rules first


def test_renames_and_hints_from_toml(tmp_path):
    toml(tmp_path / "memidx.toml", """
[[hints]]
pattern = "\\\\bfoo\\\\b"
text = "foo is gone"
[[renames]]
old = "alpha"
new = "beta"
since = "2026-01-01"
variants = ["alfa"]
""")
    core.configure(load_config(corpus=str(tmp_path), env={}))
    assert core.hints("foo") == ["foo is gone"]
    assert core.hints("alfa") == ["alpha is called beta since 2026-01-01."]
    assert core.hints("food") == []


def test_timezone(tmp_path):
    assert isinstance(today_in("UTC"), date)
    assert isinstance(today_in(""), date)
    assert isinstance(today_in("No/Such_Zone"), date)                       # falls back to UTC


def test_rezoning_without_reindex(idx):
    db = idx.build({"archive/a.md": "# A\n\nword\n"})
    assert tuple(db.execute("SELECT zone, prior FROM docs").fetchone()) == ("default", 1.0)
    idx.configure(zones=(ZoneRule("archive", 0.3, prefix="archive/"),))
    r = idx.update()
    assert r["changed"] == 0 and tuple(db.execute("SELECT zone, prior FROM docs").fetchone()) == ("archive", 0.3)


def test_exclude(idx):
    db = idx.build({"keep/a.md": "# A\n\nword\n", "skip/b.md": "# B\n\nword\n"}, exclude=("skip",))
    assert [r[0] for r in db.execute("SELECT path FROM docs")] == ["keep/a.md"]
