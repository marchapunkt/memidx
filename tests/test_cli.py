"""CLI smoke test on a temporary git repository (subprocess, like a user would run it)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def run(cwd, *args, check=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMIDX_")}
    env["MEMIDX_VECTORS"] = "0"
    p = subprocess.run([sys.executable, "-m", "memidx", *args], cwd=cwd, env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    if check:
        assert p.returncode == 0, p.stderr
    return p


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "notes").mkdir(parents=True)
    (r / "notes" / "backup.md").write_text("---\nname: Backups\ndescription: How backups work\n---\n# Backups\n\n"
                                           "## Schedule\n\nThe backup job runs nightly at 02:00 and keeps 30 snapshots.\n\n"
                                           "## Restore\n\nRestoring uses the snapshot tool. See [[network]].\n", encoding="utf-8")
    (r / "notes" / "network.md").write_text("# Network\n\nThe router forwards port 443 to the gateway.\n", encoding="utf-8")
    (r / "untracked.md").write_text("# Untracked\n\nbackup secret plans\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "add", "notes"], cwd=r, check=True)
    return r


def test_init_update_search_status_get(repo):
    p = run(repo, "init")
    assert "memidx.toml" in p.stdout and (repo / "memidx.toml").is_file()
    assert run(repo, "init", check=False).returncode == 1                    # refuses to overwrite
    p = run(repo, "update")
    assert "2 changed/new" in p.stdout                                       # only git-tracked files
    p = run(repo, "search", "backup", "snapshots")
    assert "notes/backup.md" in p.stdout and "untracked.md" not in p.stdout
    p = run(repo, "search", "--json", "-k", "1", "router")
    res = json.loads(p.stdout)
    assert res["hits"][0]["path"] == "notes/network.md" and set(res) == {"query", "hints", "hits", "tokens"}
    p = run(repo, "search", "--list", "nightly")
    assert p.stdout.startswith("1. notes/backup.md:")
    st = json.loads(run(repo, "status").stdout)
    assert st["files"] == 2 and st["config"].endswith("memidx.toml") and st["vectors"] == 0
    p = run(repo, "get", "notes/backup.md:12-20")
    assert "Restore" in p.stdout and "Schedule" not in p.stdout
    p = run(repo, "update")
    assert "0 changed/new" in p.stdout and "2 unchanged" in p.stdout


def test_search_without_index_fails_clearly(tmp_path):
    p = run(tmp_path, "search", "x", check=False)
    assert p.returncode == 2 and "memidx update" in p.stderr


def test_bad_config_fails_clearly(tmp_path):
    (tmp_path / "memidx.toml").write_text("nonsense = 1\n", encoding="utf-8")
    p = run(tmp_path, "update", check=False)
    assert p.returncode == 2 and "unknown config key" in p.stderr


def test_corpus_and_db_flags(repo, tmp_path):
    db = tmp_path / "elsewhere" / "i.db"
    run(tmp_path, "--corpus", str(repo), "--db", str(db), "update")
    assert db.is_file()
    p = run(tmp_path, "search", "--corpus", str(repo), "--db", str(db), "router")
    assert "notes/network.md" in p.stdout


def test_model_download_needs_explicit_command(repo):
    help_text = run(repo, "model", "download", "--help").stdout
    assert "download" in help_text
