"""Configuration: `memidx.toml` plus `MEMIDX_*` environment overrides.

Lookup order for the file: explicit path (`--config`), then `MEMIDX_CONFIG`, then `<corpus>/memidx.toml`,
then built-in neutral defaults. Environment variables are applied on top of the file.
"""
import fnmatch
import json
import os
import re
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone as _utc_tz
from pathlib import Path

from . import lang

CONFIG_NAME = "memidx.toml"


@dataclass(frozen=True)
class ZoneRule:
    name: str
    weight: float
    prefix: str = ""      # path prefix ("docs/") or exact path ("README.md")
    basename: str = ""    # glob on the file name ("worklog*"); checked before all prefix rules

    def matches(self, path):
        if self.basename:
            return fnmatch.fnmatchcase(os.path.basename(path), self.basename)
        p = self.prefix
        return path == p or (p.endswith("/") and path.startswith(p))


@dataclass(frozen=True)
class Ranking:
    rrf_bm25: float = 1.0        # weight of the BM25 list in reciprocal rank fusion
    rrf_vector: float = 1.0      # weight of the section-vector list
    rrf_doc: float = 0.3         # weight of the document-vector list (0 = off)
    rrf_doc_top: int = 20        # how many nearest documents contribute
    stale_factor: float = 0.6    # score factor for stale/superseded documents
    links: bool = True           # wikilink bonus on/off
    links_max: float = 0.10      # maximum wikilink bonus (0 .. 0.10)
    recency: bool = False        # optional freshness bonus (up to +15 %, half-life 180 days)


@dataclass(frozen=True)
class Config:
    corpus: str = ""
    db: str = ""
    source: str = ""                                  # file the config came from ("" = defaults)
    languages: tuple = ("de", "en")
    timezone: str = ""                                # "" = system local time
    zones: tuple = ()                                 # ZoneRule, first match wins (basename rules first)
    default_zone: ZoneRule = ZoneRule("default", 1.0)
    vector_min_weight: float = 0.7                    # zones below this weight get no vectors
    exclude: tuple = ()                               # path prefixes (directories) to skip
    hints: tuple = ()                                 # (regex on folded query, text)
    renames: tuple = ()                               # dicts: old, new, since, variants, text
    private_prefixes: tuple = ()
    own_client_prefix: str = "own:"
    private_mode: str = "own"                         # "own" or "all" (everyone sees everything)
    memory_entries_prefix: str = ""                   # "" = memory-entry rules off
    ranking: Ranking = Ranking()
    log_path: str = ""                                # "" = no query log
    vectors: bool = True
    model_cache: str = ""
    embedder: str = ""                                # "module:factory" hook, "" = built-in model
    graph_workers: int = 2
    host: str = "127.0.0.1"
    port: int = 8800
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8801
    mcp_client: str = "mcp"
    interval: int = 600

    # ---------------------------------------------------------------- derived helpers
    def zone_of(self, path):
        for z in self.zones:
            if z.basename and z.matches(path):
                return (z.name, z.weight)
        for z in self.zones:
            if not z.basename and z.matches(path):
                return (z.name, z.weight)
        return (self.default_zone.name, self.default_zone.weight)

    def all_hints(self):
        out = list(self.hints)
        for r in self.renames:
            out += rename_hints(r)
        return out

    def today(self):
        return today_in(self.timezone)


def today_in(tzname):
    if tzname:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(tzname)).date()
        except Exception:
            return datetime.now(_utc_tz.utc).date()   # unknown zone or no tz database: UTC
    return datetime.now().astimezone().date()


def default_model_cache():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "memidx", "models")


def rename_hints(r):
    """One hint per spelling (old name and each variant): "<old> is called <new> since <date>."."""
    old, new = r.get("old", ""), r.get("new", "")
    if not old or not new:
        return []
    since = r.get("since", "")
    text = r.get("text") or (f"{old} is called {new} since {since}." if since else f"{old} is now called {new}.")
    out = []
    for name in [old] + list(r.get("variants", [])):
        # alphanumeric boundaries instead of \b: \b fails when the name starts or ends with a non-word char (".corp")
        out.append((r"(?<![A-Za-z0-9])" + re.escape(lang.fold(name)) + r"(?![A-Za-z0-9])", text))
    return out


# ---------------------------------------------------------------- loading
_TOP_KEYS = {"languages", "timezone", "zones", "default_zone", "vector_min_weight", "exclude", "hints", "renames",
             "private_prefixes", "own_client_prefix", "private_mode", "memory_entries_prefix", "ranking", "log_path",
             "vectors", "model_cache", "embedder", "graph_workers", "db", "server"}
_SERVER_KEYS = {"host", "port", "mcp_host", "mcp_port", "mcp_client", "interval"}


def find_config(explicit=None, corpus=None, env=None):
    env = os.environ if env is None else env
    if explicit:
        return Path(explicit)
    if env.get("MEMIDX_CONFIG"):
        return Path(env["MEMIDX_CONFIG"])
    if corpus:
        p = Path(corpus) / CONFIG_NAME
        if p.is_file():
            return p
    return None


def _zone(d, where):
    if not isinstance(d, dict) or "name" not in d or ("prefix" in d) == ("basename" in d):
        raise ValueError(f"{where}: each zone needs 'name' and exactly one of 'prefix' or 'basename'")
    return ZoneRule(str(d["name"]), float(d.get("weight", 1.0)), str(d.get("prefix", "")), str(d.get("basename", "")))


def _from_dict(data, base, cfg_dir):
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {sorted(unknown)}")
    kw = {}
    if "languages" in data:
        kw["languages"] = lang.check_languages(data["languages"])
    for k in ("timezone", "own_client_prefix", "private_mode", "memory_entries_prefix", "embedder"):
        if k in data:
            kw[k] = str(data[k])
    if "vector_min_weight" in data:
        kw["vector_min_weight"] = float(data["vector_min_weight"])
    if "vectors" in data:
        kw["vectors"] = bool(data["vectors"])
    if "graph_workers" in data:
        kw["graph_workers"] = int(data["graph_workers"])
    for k in ("exclude", "private_prefixes"):
        if k in data:
            kw[k] = tuple(str(x) for x in data[k] if str(x))
    if "zones" in data:
        kw["zones"] = tuple(_zone(z, "zones") for z in data["zones"])
    if "default_zone" in data:
        d = data["default_zone"]
        kw["default_zone"] = ZoneRule(str(d.get("name", "default")), float(d.get("weight", 1.0)))
    if "hints" in data:
        hs = []
        for h in data["hints"]:
            re.compile(h["pattern"])
            hs.append((str(h["pattern"]), str(h["text"])))
        kw["hints"] = tuple(hs)
    if "renames" in data:
        kw["renames"] = tuple(dict(r) for r in data["renames"])
    if "ranking" in data:
        r = dict(data["ranking"])
        bad = set(r) - set(Ranking.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown ranking key(s): {sorted(bad)}")
        kw["ranking"] = replace(base.ranking, **r)
    for k in ("log_path", "model_cache", "db"):
        if k in data and data[k]:
            p = Path(os.path.expanduser(str(data[k])))
            kw[k] = str(p if p.is_absolute() or cfg_dir is None else cfg_dir / p)
    if "server" in data:
        s = dict(data["server"])
        bad = set(s) - _SERVER_KEYS
        if bad:
            raise ValueError(f"unknown server key(s): {sorted(bad)}")
        for k in ("host", "mcp_host", "mcp_client"):
            if k in s:
                kw[k] = str(s[k])
        for k in ("port", "mcp_port", "interval"):
            if k in s:
                kw[k] = int(s[k])
    return replace(base, **kw)


def _flag(v):
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def apply_env(cfg, env=None):
    """MEMIDX_* environment variables override the file (names kept from the original service)."""
    env = os.environ if env is None else env
    kw, rk = {}, {}
    if env.get("MEMIDX_LANGUAGES"):
        kw["languages"] = lang.check_languages(env["MEMIDX_LANGUAGES"].split(","))
    if env.get("MEMIDX_TIMEZONE"):
        kw["timezone"] = env["MEMIDX_TIMEZONE"]
    if env.get("MEMIDX_ZONES") and os.path.exists(env["MEMIDX_ZONES"]):
        with open(env["MEMIDX_ZONES"], encoding="utf-8") as f:
            kw["zones"] = tuple(ZoneRule(z[1], float(z[2]), prefix=z[0]) for z in json.load(f))
    if env.get("MEMIDX_TOMBSTONES") and os.path.exists(env["MEMIDX_TOMBSTONES"]):
        with open(env["MEMIDX_TOMBSTONES"], encoding="utf-8") as f:
            kw["hints"] = tuple((t[0], t[1]) for t in json.load(f))
        kw["renames"] = ()                       # the original replaced the whole hint list
    if "MEMIDX_EXCLUDE" in env:
        kw["exclude"] = tuple(d for d in env["MEMIDX_EXCLUDE"].split(",") if d)
    if "MEMIDX_PRIVATE_PREFIXES" in env:
        kw["private_prefixes"] = tuple(p for p in env["MEMIDX_PRIVATE_PREFIXES"].split(",") if p)
    if env.get("MEMIDX_OWN_PREFIX"):
        kw["own_client_prefix"] = env["MEMIDX_OWN_PREFIX"]
    if env.get("MEMIDX_PRIVATE_MODE"):
        kw["private_mode"] = env["MEMIDX_PRIVATE_MODE"]
    for name in ("MEMIDX_MEMORY_PREFIX", "MEMIDX_GEDAECHTNIS_PREFIX"):
        if name in env:
            kw["memory_entries_prefix"] = env[name]
    if "MEMIDX_LOG" in env:
        kw["log_path"] = env["MEMIDX_LOG"]
    if "MEMIDX_VECTORS" in env:
        kw["vectors"] = _flag(env["MEMIDX_VECTORS"])
    if env.get("MEMIDX_MODEL_CACHE"):
        kw["model_cache"] = env["MEMIDX_MODEL_CACHE"]
    if env.get("MEMIDX_EMBEDDER"):
        kw["embedder"] = env["MEMIDX_EMBEDDER"]
    if env.get("MEMIDX_GRAPH_WORKERS"):
        kw["graph_workers"] = int(env["MEMIDX_GRAPH_WORKERS"])
    if env.get("MEMIDX_DB"):
        kw["db"] = env["MEMIDX_DB"]
    if env.get("MEMIDX_MCP_HOST"):
        kw["mcp_host"] = env["MEMIDX_MCP_HOST"]
    if env.get("MEMIDX_MCP_PORT"):
        kw["mcp_port"] = int(env["MEMIDX_MCP_PORT"])
    if env.get("MEMIDX_MCP_CLIENT"):
        kw["mcp_client"] = env["MEMIDX_MCP_CLIENT"]
    if env.get("MEMIDX_INTERVAL"):
        kw["interval"] = int(env["MEMIDX_INTERVAL"])
    if "MEMIDX_RECENCY" in env:
        rk["recency"] = env["MEMIDX_RECENCY"] == "1"
    if "MEMIDX_LINKS" in env:
        rk["links"] = env["MEMIDX_LINKS"] != "0"
    if env.get("MEMIDX_LINKS_MAX"):
        try:
            rk["links_max"] = float(env["MEMIDX_LINKS_MAX"])
        except ValueError:
            pass
    for name, key, typ in (("MEMIDX_RRF_BM", "rrf_bm25", float), ("MEMIDX_RRF_VEC", "rrf_vector", float),
                           ("MEMIDX_RRF_DOC", "rrf_doc", float), ("MEMIDX_RRF_DOCTOP", "rrf_doc_top", int)):
        if env.get(name):
            rk[key] = typ(env[name])
    if rk:
        kw["ranking"] = replace(cfg.ranking, **rk)
    return replace(cfg, **kw)


def load_config(path=None, corpus=None, db=None, env=None):
    """Build the effective configuration. `corpus` and `db` are explicit values (CLI flags) and win."""
    env = os.environ if env is None else env
    corpus = corpus or env.get("MEMIDX_CORPUS") or os.getcwd()
    found = find_config(path, corpus, env)
    cfg = Config()
    if found is not None:
        with open(found, "rb") as f:
            data = tomllib.load(f)
        cfg = _from_dict(data, cfg, found.parent.resolve())
        cfg = replace(cfg, source=str(found))
    cfg = apply_env(cfg, env)
    kw = {"corpus": str(Path(corpus).resolve())}
    if db:
        kw["db"] = str(db)
    elif not cfg.db:
        kw["db"] = str(Path(corpus).resolve() / ".memidx" / "index.db")
    if not cfg.model_cache:
        kw["model_cache"] = default_model_cache()
    return replace(cfg, **kw)


EXAMPLE = '''# memidx configuration. All keys are optional; this file shows the defaults or commented examples.
# Lookup: --config, then $MEMIDX_CONFIG, then <corpus>/memidx.toml. MEMIDX_* environment variables override it.

# Languages for full-text normalization: "de", "en" or both. Changing this rebuilds the full-text index on
# the next `memidx update`.
languages = ["de", "en"]

# Time zone for "today" (validity dates of memory entries). "" = system local time.
timezone = ""

# Where the index lives. Relative paths are relative to this file. Default: <corpus>/.memidx/index.db
# db = ".memidx/index.db"

# Directory prefixes that are not indexed.
exclude = []

# Zones group files and weight their scores. First match wins; `basename` rules are checked before
# `prefix` rules. A prefix ending in "/" matches a directory, otherwise it matches one exact path.
# Files that match nothing get the default zone.
[default_zone]
name = "default"
weight = 1.0

# [[zones]]
# prefix = "notes/"
# name = "notes"
# weight = 1.0
#
# [[zones]]
# prefix = "archive/"
# name = "archive"
# weight = 0.3
#
# [[zones]]
# basename = "worklog*"      # file-name glob, e.g. per-task scratch files
# name = "scratch"
# weight = 0.6

# Zones with a weight below this value are searched by full text only (no vectors).
vector_min_weight = 0.7

# Hints are shown before the hits when the (folded, lower-case) query matches the regex.
# [[hints]]
# pattern = "\\\\boldwiki\\\\b"
# text = "OldWiki was shut down; the notes now live under notes/."

# Renames generate the hint "<old> is called <new> since <since>." for the old name and every variant.
# [[renames]]
# old = "oldname"
# new = "newname"
# since = "2026-01-31"
# variants = ["old-name"]

# Paths under these prefixes are only visible to clients whose self-declared client id starts with
# own_client_prefix. This is a convention against accidental disclosure, NOT access control.
private_prefixes = []
own_client_prefix = "own:"
private_mode = "own"          # "all" = everyone sees private paths

# Files under this prefix are memory entries with their own validity rules ("" = off).
memory_entries_prefix = ""

# Append one JSON line per query from an identified client to this file ("" = off).
log_path = ""

# Vectors need `pip install memidx[vectors]` and `memidx model download`. false = full text only.
vectors = true
# model_cache = "~/.cache/memidx/models"

[ranking]
rrf_bm25 = 1.0
rrf_vector = 1.0
rrf_doc = 0.3
rrf_doc_top = 20
stale_factor = 0.6
links = true
links_max = 0.10
recency = false

[server]
host = "127.0.0.1"
port = 8800
mcp_host = "127.0.0.1"
mcp_port = 8801
mcp_client = "mcp"
interval = 600
'''
