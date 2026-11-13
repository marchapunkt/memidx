"""The documents must name the version the package has, so a release cannot ship a stale README."""
import re
import tomllib
from pathlib import Path

import memidx

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_the_same_everywhere():
    version = memidx.__version__
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == version
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^\*\*Version " + re.escape(version) + r"\*\*", readme, re.M), "README.md does not state the version"
    first = re.search(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert first and first.group(1) == version, "the newest CHANGELOG.md entry is not this version"
