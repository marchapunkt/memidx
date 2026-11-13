"""Security hardening of 0.2.0: trust levels of config files (F1), index location and file modes (F2),
environment overrides (F3), explicit empty results and exit codes (F4), git invocation (F5), opt-in vectors
with explicit model presets (F6). The subprocess tests run the real CLI entry with cwd = corpus."""
import contextlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from memidx import cli, config, core, embed, server
from memidx.config import CORPUS_KEYS, load_config, user_config_path

POSIX = os.name == "posix"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
ROOT = Path(__file__).resolve().parents[1]


def run(cwd, *args, env_extra=None, check=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMIDX_")}
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, "-m", "memidx", *args], cwd=cwd, env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    if check is not None:
        assert p.returncode == check, (p.returncode, p.stdout, p.stderr)
    return p


def main(*argv):
    """In-process CLI call: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def toml_str(p):
    return json.dumps(str(p))          # a JSON string is a valid TOML basic string (escapes Windows backslashes)


@pytest.fixture
def corpus(tmp_path):
    c = tmp_path / "corpus"
    write(c / "notes" / "backup.md", "# Backups\n\nThe backup job runs nightly and keeps snapshots.\n")
    write(c / "notes" / "network.md", "# Network\n\nThe router forwards port 443.\n")
    write(c / "private" / "diary.md", "# Diary\n\nquokkasecret lives here, backup too.\n")
    return c


@pytest.fixture
def repo(corpus):
    subprocess.run(["git", "init", "-q"], cwd=corpus, check=True)
    subprocess.run(["git", "add", "notes", "private"], cwd=corpus, check=True)
    return corpus


def files_under(d):
    return sorted(str(p.relative_to(d)) for p in Path(d).rglob("*")) if Path(d).exists() else []


# ====================================================================== F1: config trust levels
EVILMOD = '''import pathlib
pathlib.Path(__file__).with_name("MARKER").write_text("code from the repository ran")


class E:
    name = "evil"
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]
    def embed_query(self, text):
        return [1.0, 0.0]


def make(cfg=None):
    return E()
'''


@needs_git
def test_corpus_embedder_is_never_imported(repo):
    write(repo / "evilmod.py", EVILMOD)
    write(repo / "memidx.toml", 'embedder = "evilmod:make"\nvectors = true\n')
    env = {"MEMIDX_VECTORS": "1", "MEMIDX_MODEL": "embeddinggemma-300m"}   # vectors on: the hook would be used
    p = run(repo, "update", env_extra=env, check=0)
    assert "embedder" in p.stderr and "user config" in p.stderr and p.stderr.count("ignored key(s)") == 1
    p = run(repo, "search", "backup", env_extra=env, check=0)
    assert "notes/backup.md" in p.stdout and "embedder" in p.stderr
    assert not (repo / "MARKER").exists()


@needs_git
def test_same_file_as_explicit_config_is_trusted(repo):
    """Control for the test above: the same file passed with --config is trusted and the hook runs."""
    write(repo / "evilmod.py", EVILMOD)
    write(repo / "memidx.toml", 'embedder = "evilmod:make"\nvectors = true\n')
    run(repo, "--config", str(repo / "memidx.toml"), "update", check=0)
    assert (repo / "MARKER").exists()


@needs_git
def test_corpus_db_and_log_path_outside_are_ignored(repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    write(repo / "memidx.toml", f"db = {toml_str(outside / 'evil.db')}\nlog_path = {toml_str(outside / 'q.jsonl')}\n"
                                f"model_cache = {toml_str(outside / 'models')}\n")
    p = run(repo, "update", check=0)
    assert "db" in p.stderr and "log_path" in p.stderr and "model_cache" in p.stderr
    run(repo, "search", "backup", "--client", "someone", check=0)
    run(repo, "status", check=0)
    assert files_under(outside) == []


def test_corpus_private_mode_all_is_ignored(corpus, user_dirs):
    write(user_config_path(), 'private_prefixes = ["private/"]\n')
    write(corpus / "memidx.toml", 'private_mode = "all"\nprivate_prefixes = []\nown_client_prefix = ""\n')
    cfg = load_config(corpus=str(corpus))
    assert cfg.private_mode == "own" and cfg.private_prefixes == ("private/",) and cfg.own_client_prefix == "own:"
    assert len(cfg.notices) == 1 and "private_mode" in cfg.notices[0] and "private_prefixes" in cfg.notices[0]


def test_corpus_file_allowed_keys_still_apply(corpus):
    write(corpus / "memidx.toml", 'timezone = "UTC"\nexclude = ["skip"]\n[[zones]]\nprefix = "notes/"\nname = "n"\n'
                                  'weight = 0.5\n[ranking]\nlinks = false\n')
    cfg = load_config(corpus=str(corpus))
    assert cfg.timezone == "UTC" and cfg.exclude == ("skip",) and cfg.zone_of("notes/a.md") == ("n", 0.5)
    assert cfg.ranking.links is False and cfg.notices == ()
    assert cfg.sources == ((str(corpus / "memidx.toml"), "corpus"),)


def test_every_other_key_is_ignored_with_one_notice(corpus, tmp_path):
    lines = ['embedder = "x:y"', f"db = {toml_str(tmp_path / 'x.db')}", f"log_path = {toml_str(tmp_path / 'l')}",
             f"model_cache = {toml_str(tmp_path / 'm')}", "vectors = true", 'model = "embeddinggemma-300m"',
             'private_prefixes = ["a/"]', 'own_client_prefix = ""', 'private_mode = "all"', "graph_workers = 9",
             "git_safe_directory = true", "totally_unknown = 1", "[server]", 'host = "0.0.0.0"', "port = 1"]
    write(corpus / "memidx.toml", "\n".join(lines) + "\n")
    cfg = load_config(corpus=str(corpus))
    d = config.Config()
    for f in ("embedder", "log_path", "vectors", "model", "private_prefixes", "own_client_prefix", "private_mode",
              "graph_workers", "git_safe_directory", "host", "port"):
        assert getattr(cfg, f) == getattr(d, f), f
    assert cfg.db == config.default_db_path(corpus) and cfg.model_cache == config.default_model_cache()
    assert len(cfg.notices) == 1
    for k in ("embedder", "db", "log_path", "model_cache", "vectors", "model", "private_prefixes", "own_client_prefix",
              "private_mode", "graph_workers", "git_safe_directory", "server", "totally_unknown"):
        assert k in cfg.notices[0], k
    assert str(user_config_path()) in cfg.notices[0] and "--config" in cfg.notices[0]


def test_corpus_keys_are_exactly_the_documented_set():
    assert CORPUS_KEYS == {"languages", "timezone", "exclude", "default_zone", "zones", "vector_min_weight", "hints",
                           "renames", "memory_entries_prefix", "ranking"}


def test_layering_order(corpus, tmp_path):
    write(user_config_path(), 'timezone = "user"\nlanguages = ["en"]\nlog_path = "u.log"\n')
    assert load_config(corpus=str(corpus)).timezone == "user"
    write(corpus / "memidx.toml", 'timezone = "corpus"\n')
    cfg = load_config(corpus=str(corpus))
    assert cfg.timezone == "corpus" and cfg.languages == ("en",)                  # user keys stay below
    assert [k for _, k in cfg.sources] == ["user", "corpus"]
    explicit = write(tmp_path / "explicit.toml", 'timezone = "explicit"\n')
    assert load_config(str(explicit), corpus=str(corpus)).timezone == "explicit"
    assert load_config(corpus=str(corpus), env={**os.environ, "MEMIDX_CONFIG": str(explicit)}).timezone == "explicit"
    env = {**os.environ, "MEMIDX_CONFIG": str(explicit), "MEMIDX_TIMEZONE": "env", "MEMIDX_DB": str(tmp_path / "env.db")}
    cfg = load_config(corpus=str(corpus), env=env)
    assert cfg.timezone == "env" and cfg.db == str(tmp_path / "env.db")
    cfg = load_config(corpus=str(corpus), env=env, db=str(tmp_path / "flag.db"))
    assert cfg.db == str(tmp_path / "flag.db") and "MEMIDX_DB" not in cfg.env_used   # the flag wins


def test_no_corpus_config_flag_and_env(corpus):
    write(corpus / "memidx.toml", 'timezone = "corpus"\nembedder = "x:y"\n')
    assert load_config(corpus=str(corpus), corpus_config=False).timezone == ""
    cfg = load_config(corpus=str(corpus), env={**os.environ, "MEMIDX_NO_CORPUS_CONFIG": "1"})
    assert cfg.timezone == "" and cfg.notices == () and cfg.sources == ()
    rc, out, err = main("--corpus", str(corpus), "--no-corpus-config", "status")
    assert rc == 3 and json.loads(out)["config"] == [] and "ignored key" not in err


def test_init_in_corpus_writes_only_allowed_keys(corpus, monkeypatch):
    monkeypatch.chdir(corpus)
    rc, out, _ = main("init")
    text = (corpus / "memidx.toml").read_text(encoding="utf-8")
    assert rc == 0 and str(user_config_path()) in text
    import tomllib
    assert set(tomllib.loads(text)) <= CORPUS_KEYS
    assert load_config(corpus=str(corpus)).notices == ()


def test_init_user_writes_full_example(corpus, monkeypatch):
    monkeypatch.chdir(corpus)
    rc, out, _ = main("init", "--user")
    p = user_config_path()
    assert rc == 0 and p.is_file() and not (corpus / "memidx.toml").exists()
    import tomllib
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert {"server", "private_prefixes", "vectors", "git_safe_directory", "log_path"} <= set(data)
    assert load_config(corpus=str(corpus)).sources == ((str(p), "user"),)
    if POSIX:
        assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700


def test_untrusted_hints_are_compiled_defensively(corpus):
    long_pat = "a" * (config.MAX_PATTERN_CHARS + 1)
    write(corpus / "memidx.toml", f'''
[[hints]]
pattern = "(unclosed"
text = "broken"
[[hints]]
pattern = "{long_pat}"
text = "too long"
[[hints]]
pattern = "good"
text = "kept"
[[hints]]
text = "no pattern"
[[renames]]
old = "{long_pat}"
new = "x"
[[renames]]
old = "alpha"
new = "beta"
''')
    cfg = core.configure(load_config(corpus=str(corpus)))
    assert cfg.hints == (("good", "kept"),) and [r["old"] for r in cfg.renames] == ["alpha"]
    assert core.hints("good alpha") == ["kept", "alpha is now called beta."]
    assert any("skipped invalid hint" in n for n in cfg.notices) and any("skipped invalid rename" in n for n in cfg.notices)


def test_untrusted_hint_count_is_capped(corpus):
    n = config.MAX_PATTERNS + 20
    write(corpus / "memidx.toml", "".join(f'[[hints]]\npattern = "w{i}"\ntext = "t"\n' for i in range(n)))
    cfg = load_config(corpus=str(corpus))
    assert len(cfg.hints) == config.MAX_PATTERNS and any("only the first" in x for x in cfg.notices)


def test_trusted_hints_stay_strict(tmp_path, corpus):
    p = write(tmp_path / "t.toml", '[[hints]]\npattern = "(unclosed"\ntext = "x"\n')
    with pytest.raises(Exception):
        load_config(str(p), corpus=str(corpus))


# ====================================================================== F2: index location and modes
@needs_git
def test_default_index_is_outside_the_corpus(repo, user_dirs):
    p = run(repo, "update", check=0)
    db = config.default_db_path(repo)
    assert os.path.isfile(db) and not config.is_inside(db, repo)
    assert Path(db).is_relative_to(Path(user_dirs["LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"]).resolve()) \
        or Path(db).is_relative_to(Path(user_dirs["XDG_CACHE_HOME"]))
    name = Path(db).parent.name
    assert name.startswith("corpus-") and len(name) == len("corpus-") + 16
    assert not (repo / ".memidx").exists() and "git-tracked files" in p.stdout
    st = json.loads(run(repo, "status", check=0).stdout)
    assert st["db"] == db


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX; Windows relies on the user profile ACL")
def test_posix_modes_of_index_and_log(corpus, tmp_path):
    old = os.umask(0o022)
    try:
        log = tmp_path / "logs" / "q.jsonl"
        (tmp_path / "logs").mkdir()
        write(user_config_path(), f"log_path = {toml_str(log)}\n")
        cfg = core.configure(load_config(corpus=str(corpus)))
        core.update(cfg.corpus, cfg.db, log=lambda m: None)
        d = Path(cfg.db).parent
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
        assert stat.S_IMODE(d.parent.stat().st_mode) == 0o700                    # .../memidx/index
        for f in (cfg.db, cfg.db + "-wal", cfg.db + "-shm"):
            if os.path.exists(f):
                assert stat.S_IMODE(os.stat(f).st_mode) == 0o600, f
        db = core.open_index(cfg.db)
        core.search(db, "backup", client="tester")
        db.close()
        assert stat.S_IMODE(log.stat().st_mode) == 0o600
    finally:
        os.umask(old)


def test_legacy_index_notice_without_touching_it(corpus):
    legacy = write(corpus / ".memidx" / "index.db", "old index")
    before = legacy.stat().st_mtime_ns
    cfg = load_config(corpus=str(corpus))
    assert any("no longer used" in n and str(legacy) in n for n in cfg.notices)
    core.configure(cfg)
    core.update(cfg.corpus, cfg.db, log=lambda m: None)
    assert legacy.read_text(encoding="utf-8") == "old index" and legacy.stat().st_mtime_ns == before
    explicit = load_config(corpus=str(corpus), db=str(corpus.parent / "x.db"))
    assert not any("no longer used" in n for n in explicit.notices)


# ====================================================================== F3: environment overrides
@needs_git
def test_memidx_url_no_longer_redirects_search(repo):
    run(repo, "update", check=0)
    p = run(repo, "search", "backup", env_extra={"MEMIDX_URL": "http://127.0.0.1:9"}, check=0)
    assert "notes/backup.md" in p.stdout and "not reachable" not in p.stderr


@needs_git
def test_isolated_ignores_memidx_db_and_user_config(repo, tmp_path):
    run(repo, "update", check=0)                                   # default location
    elsewhere = tmp_path / "env.db"
    env = {"MEMIDX_DB": str(elsewhere)}
    p = run(repo, "search", "backup", env_extra=env)
    assert p.returncode == 3 and not elsewhere.exists()             # MEMIDX_DB is honored without --isolated
    notice = [l for l in p.stderr.splitlines() if "environment variable" in l]
    assert len(notice) == 1 and "MEMIDX_DB" in notice[0] and str(elsewhere) not in notice[0]
    p = run(repo, "--isolated", "search", "backup", env_extra=env, check=0)
    assert "notes/backup.md" in p.stdout and "MEMIDX_DB" not in p.stderr
    write(user_config_path(), f"db = {toml_str(tmp_path / 'user.db')}\n")
    assert run(repo, "search", "backup").returncode == 3
    run(repo, "search", "-I", "backup", check=0)


@needs_git
def test_status_lists_env_names_not_values(repo):
    run(repo, "update", check=0)
    secret = "value-that-must-not-appear"
    env = {"MEMIDX_TIMEZONE": secret, "MEMIDX_URL": secret, "MEMIDX_CLIENT": secret}
    p = run(repo, "status", env_extra=env, check=0)
    st = json.loads(p.stdout)
    assert secret not in p.stdout and secret not in p.stderr
    assert st["environment"]["set"] == ["MEMIDX_CLIENT", "MEMIDX_TIMEZONE", "MEMIDX_URL"]
    assert st["environment"]["effective"] == ["MEMIDX_TIMEZONE"]
    st = json.loads(run(repo, "-I", "status", env_extra=env, check=0).stdout)
    assert st["environment"]["effective"] == [] and st["isolated"] is True


@needs_git
def test_location_env_notice_names_only(repo, tmp_path):
    log = tmp_path / "secret-dir" / "q.jsonl"
    env = {"MEMIDX_LOG": str(log), "MEMIDX_MODEL_CACHE": str(tmp_path / "secret-models")}
    for cmd in (("update",), ("search", "backup")):
        p = run(repo, *cmd, env_extra=env, check=0)
        lines = [l for l in p.stderr.splitlines() if "environment variable" in l]
        assert len(lines) == 1 and "MEMIDX_LOG" in lines[0] and "MEMIDX_MODEL_CACHE" in lines[0]
        assert "secret-dir" not in lines[0] and "secret-models" not in lines[0]
    assert "environment variable" not in run(repo, "search", "backup", check=0).stderr


# ====================================================================== F4: explicit empty results
def test_search_on_missing_index_exits_3_and_creates_nothing(corpus, tmp_path):
    db = tmp_path / "nothere" / "index.db"
    for cmd in (("search", "backup"), ("status",), ("get", "notes/backup.md"), ("mcp",)):
        rc, out, err = main("--corpus", str(corpus), "--db", str(db), *cmd)
        assert rc == 3 and "memidx update" in err, cmd
    assert not db.parent.exists()
    default = Path(config.default_db_path(corpus))
    rc, _, _ = main("--corpus", str(corpus), "search", "backup")
    assert rc == 3 and not default.exists() and not default.parent.exists()


def test_empty_or_schemaless_index_exits_3(corpus, tmp_path):
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    rc, _, err = main("--corpus", str(corpus), "--db", str(empty), "search", "backup")
    assert rc == 3 and "schema" in err and empty.stat().st_size == 0
    emptycorpus = tmp_path / "nothing"
    emptycorpus.mkdir()
    db = tmp_path / "e.db"
    assert main("--corpus", str(emptycorpus), "--db", str(db), "update")[0] == 0
    rc, _, err = main("--corpus", str(emptycorpus), "--db", str(db), "search", "x")
    assert rc == 3 and "empty" in err


def test_normalization_mismatch_needs_update(corpus, tmp_path):
    db = tmp_path / "i.db"
    en = write(tmp_path / "en.toml", 'languages = ["en"]\n')
    assert main("--corpus", str(corpus), "--db", str(db), "--config", str(en), "update")[0] == 0
    assert main("--corpus", str(corpus), "--db", str(db), "--config", str(en), "search", "backup")[0] == 0
    rc, _, err = main("--corpus", str(corpus), "--db", str(db), "search", "backup")
    assert rc == 3 and "normalization" in err and "memidx update" in err


@pytest.fixture
def built(corpus, tmp_path):
    db = tmp_path / "i.db"
    user = write(user_config_path(), 'private_prefixes = ["private/"]\n')
    assert main("--corpus", str(corpus), "--db", str(db), "update")[0] == 0
    return corpus, db


def test_exit_codes_hit_no_match_empty_query_unknown_zone(built):
    corpus, db = built
    base = ("--corpus", str(corpus), "--db", str(db), "search")
    rc, out, err = main(*base, "--json", "backup")
    assert rc == 0 and json.loads(out)["hits"] and "reason" not in json.loads(out)
    rc, out, err = main(*base, "--json", "zzzunknownword")
    res = json.loads(out)
    assert rc == 1 and res["hits"] == [] and res["reason"] == "no_match" and "git grep" in res["message"]
    rc, out, err = main(*base, "--json", "the", "and", "of")
    res = json.loads(out)
    assert rc == 2 and res["reason"] == "empty_query" and res["dropped"] == ["the", "and", "of"]
    assert "the" in err and "dropped" in err
    rc, out, err = main(*base, "--json", "--zone", "nosuchzone", "backup")
    res = json.loads(out)
    assert rc == 2 and res["reason"] == "unknown_zone" and res["zones"] == ["default"] and "default" in err
    rc, out, err = main(*base, "zzzunknownword")
    assert rc == 1 and "(no hits) [no_match]" in out


def test_private_excluded_flag_reveals_nothing(built):
    corpus, db = built
    base = ("--corpus", str(corpus), "--db", str(db), "search", "--json")
    for q in ("quokkasecret", "zzzunknownword"):           # matches only privately / matches nothing: same answer
        rc, out, err = main(*base, q)
        res = json.loads(out)
        assert rc == 1 and res["private_excluded"] is True and res["reason"] == "no_match"
        assert err.count("private paths were not searched") == 1
    rc, out, err = main(*base, "--client", "own:me", "quokkasecret")
    assert rc == 0 and "private_excluded" not in json.loads(out) and "private paths" not in err


def test_no_private_flag_without_prefixes(corpus, tmp_path):
    db = tmp_path / "i.db"
    main("--corpus", str(corpus), "--db", str(db), "update")
    rc, out, err = main("--corpus", str(corpus), "--db", str(db), "search", "--json", "backup")
    assert rc == 0 and "private_excluded" not in json.loads(out) and "private paths" not in err


@needs_git
def test_git_failure_in_work_tree_is_an_error(corpus, tmp_path):
    (corpus / ".git").write_text("gitdir: " + str(tmp_path / "no-such-gitdir") + "\n", encoding="utf-8")
    db = tmp_path / "i.db"
    p = run(corpus, "--db", str(db), "update")
    assert p.returncode == 3 and "git ls-files" in p.stderr and not db.exists()


def test_tree_walk_outside_git_says_so(corpus, tmp_path):
    rc, out, _ = main("--corpus", str(corpus), "--db", str(tmp_path / "i.db"), "update")
    assert rc == 0 and "all *.md files, not a git work tree" in out


def test_readonly_open_never_creates(tmp_path):
    p = tmp_path / "missing.db"
    import sqlite3
    with pytest.raises(sqlite3.Error):
        core.open_db(str(p), readonly=True)
    with pytest.raises(sqlite3.Error):
        core.open_db(str(p))
    assert core.index_problem(str(p))[0] == "missing" and not p.exists()


def _http(handler_arg):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(handler_arg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def get(path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}{path}", timeout=30) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")
    return srv, get


def test_http_reasons_and_status_codes(built, tmp_path):
    corpus, db = built
    core.configure(load_config(corpus=str(corpus), db=str(db)))
    srv, get = _http(str(db))
    try:
        s, body = get("/search?q=zzzunknownword")
        assert s == 200 and json.loads(body)["reason"] == "no_match" and json.loads(body)["private_excluded"] is True
        s, body = get("/search?q=" + urllib.parse.quote("the and"))
        assert s == 400 and json.loads(body)["reason"] == "empty_query"
        s, body = get("/search?q=backup&zone=nope")
        assert s == 400 and json.loads(body)["reason"] == "unknown_zone"
        s, body = get("/search?q=zzzunknownword&mode=text")
        assert s == 200 and "[no_match]" in body
    finally:
        srv.shutdown(); srv.server_close()
    missing = tmp_path / "missing.db"
    srv, get = _http(str(missing))
    try:
        for path in ("/search?q=backup", "/status", "/"):
            s, body = get(path)
            assert s == 503 and json.loads(body)["reason"] == "no_index", path
        assert not missing.exists()
    finally:
        srv.shutdown(); srv.server_close()


def test_remote_search_maps_status_to_exit_codes(built):
    corpus, db = built
    core.configure(load_config(corpus=str(corpus), db=str(db)))
    srv, _ = _http(str(db))
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert main("search", "--url", url, "backup")[0] == 0
        assert main("search", "--url", url, "zzzunknownword")[0] == 1
        assert main("search", "--url", url, "the")[0] == 2
    finally:
        srv.shutdown(); srv.server_close()
    assert main("search", "--url", "http://127.0.0.1:9", "backup")[0] == 3


def test_mcp_text_includes_reason(built, tmp_path):
    pytest.importorskip("mcp")
    import asyncio
    from memidx import mcp_server
    from test_mcp import _text
    corpus, db = built
    cfg = core.configure(load_config(corpus=str(corpus), db=str(db)))
    mcp = mcp_server.build_server(cfg)
    out = _text(asyncio.run(mcp.call_tool("memory_search", {"query": "zzzunknownword"})))
    assert "[no_match]" in out and "private paths were not searched" in out
    out = _text(asyncio.run(mcp.call_tool("memory_search", {"query": "the"})))
    assert "[empty_query]" in out
    from dataclasses import replace
    mcp = mcp_server.build_server(replace(cfg, db=str(tmp_path / "missing.db")))
    out = _text(asyncio.run(mcp.call_tool("memory_search", {"query": "backup"})))
    assert "[no_index]" in out and not (tmp_path / "missing.db").exists()


def test_readme_has_pointer_not_proof_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## A pointer, not proof" in text and "git grep" in text


# ====================================================================== F5: git invocation
class _Capture:
    def __init__(self, returncode=0, stdout=b"notes/a.md\0", stderr=b""):
        self.calls = []
        self.result = (returncode, stdout, stderr)

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, self.result[0], self.result[1], self.result[2])


def test_git_argv_without_safe_directory_by_default(corpus, monkeypatch):
    (corpus / ".git").mkdir()
    cap = _Capture()
    monkeypatch.setattr(core.subprocess, "run", cap)
    core.configure(load_config(corpus=str(corpus)))
    files, mode = core.md_files(str(corpus))
    argv, kw = cap.calls[0]
    assert files == ["notes/a.md"] and mode == core.GIT_MODE
    assert not any("safe.directory" in a for a in argv)
    assert "core.fsmonitor=false" in argv and "core.untrackedCache=false" in argv
    assert kw["env"]["GIT_OPTIONAL_LOCKS"] == "0"


def test_safe_directory_only_from_trusted_sources(corpus, monkeypatch, tmp_path):
    (corpus / ".git").mkdir()
    cap = _Capture()
    monkeypatch.setattr(core.subprocess, "run", cap)
    write(corpus / "memidx.toml", "git_safe_directory = true\n")
    cfg = load_config(corpus=str(corpus))
    assert cfg.git_safe_directory is False and "git_safe_directory" in cfg.notices[0]
    for cfg in (load_config(corpus=str(corpus), trust_repo_owner=True),
                load_config(corpus=str(corpus), env={**os.environ, "MEMIDX_GIT_SAFE_DIRECTORY": "1"}),
                load_config(str(write(tmp_path / "t.toml", "git_safe_directory = true\n")), corpus=str(corpus))):
        core.configure(cfg)
        core.md_files(str(corpus))
        assert "safe.directory=*" in cap.calls[-1][0]


def test_dubious_ownership_names_the_setting(corpus, monkeypatch, tmp_path):
    (corpus / ".git").mkdir()
    monkeypatch.setattr(core.subprocess, "run", _Capture(128, b"", b"fatal: detected dubious ownership in repository"))
    db = tmp_path / "i.db"
    rc, out, err = main("--corpus", str(corpus), "--db", str(db), "update")
    assert rc == 3 and "git_safe_directory" in err and "--trust-repo-owner" in err and not db.exists()


def test_dockerfile_trusts_mounted_repository():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "MEMIDX_GIT_SAFE_DIRECTORY=1" in text and "MEMIDX_URL" not in text


# ====================================================================== F6: opt-in vectors, explicit model
def test_vectors_off_and_no_model_by_default(corpus):
    cfg = load_config(corpus=str(corpus))
    assert cfg.vectors is False and cfg.model == "" and cfg.embedder == ""


def test_vectors_without_model_exit_2(corpus, tmp_path):
    write(user_config_path(), "vectors = true\n")
    for cmd in (("update",), ("search", "backup"), ("serve", "--no-auto-update"), ("mcp",)):
        rc, _, err = main("--corpus", str(corpus), "--db", str(tmp_path / "i.db"), *cmd)
        assert rc == 2 and "no model is configured" in err and "memidx model list" in err, cmd
    assert not (tmp_path / "i.db").exists()


def test_unknown_model_preset_is_a_config_error(corpus, tmp_path):
    write(user_config_path(), 'vectors = true\nmodel = "nosuch"\n')
    rc, _, err = main("--corpus", str(corpus), "--db", str(tmp_path / "i.db"), "update")
    assert rc == 2 and "unknown model preset" in err


def test_model_download_of_non_osi_preset_needs_accept_license(monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(embed, "download_model", forbidden)
    hub = type(sys)("huggingface_hub")
    hub.snapshot_download = forbidden
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    rc, out, err = main("model", "download", "embeddinggemma-300m", "--cache-dir", str(tmp_path / "m"))
    assert rc == 2 and "Gemma Terms of Use" in err and "https://ai.google.dev/gemma/terms" in err
    assert "NOT an OSI-approved" in err and "--accept-license" in err and not (tmp_path / "m").exists()
    with pytest.raises(SystemExit):                                  # the preset name is required
        main("model", "download")
    assert main("model", "download", "nosuch")[0] == 2


def test_model_download_with_accept_license_calls_downloader(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(embed, "download_model", lambda cache, preset, log=print: seen.append((cache, preset.name)))
    rc, _, _ = main("model", "download", "embeddinggemma-300m", "--accept-license", "--cache-dir", str(tmp_path))
    assert rc == 0 and seen == [(str(tmp_path), "embeddinggemma-300m")]


def test_model_list_shows_license_and_osi_status():
    rc, out, _ = main("model", "list")
    assert rc == 0 and "embeddinggemma-300m" in out and "Gemma Terms of Use" in out and "OSI-approved: NO" in out
    assert "ships no model" in out


def test_preset_table_structure():
    p = embed.PRESETS["embeddinggemma-300m"]
    for f in ("repository", "revision", "files", "query_prompt", "document_prompt", "dimension", "max_tokens",
              "license_name", "license_url"):
        assert getattr(p, f), f
    assert p.osi_approved is False and all(len(v) == 2 and len(v[1]) == 64 for v in p.files.values())
    # the embedder name stored in 0.1.0 indexes is unchanged, so existing vectors stay valid
    assert p.embedder_name == "onnx-community/embeddinggemma-300m-ONNX@5090578d9565/onnx/model_q4.onnx"
