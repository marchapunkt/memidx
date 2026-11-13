"""Configuration: TOML files with two trust levels, plus `MEMIDX_*` environment overrides and CLI flags.

Trusted sources may set every key:
  * the per-user config (`$XDG_CONFIG_HOME/memidx/config.toml` or `~/.config/memidx/config.toml`;
    on Windows `%APPDATA%\\memidx\\config.toml`),
  * an explicit file (`--config PATH` or `MEMIDX_CONFIG`).
Untrusted source: `<corpus>/memidx.toml`. The indexed repository may come from someone else, so only the keys
in CORPUS_KEYS are honored there (zones, languages, hints ...). Every other key is ignored with one notice.

Layering, lowest first: built-in defaults < user config < corpus file (allowed keys only) < explicit config
< environment variables < CLI flags. `isolated=True` (`--isolated`) ignores all `MEMIDX_*` variables and the
user config; `corpus_config=False` (`--no-corpus-config`, `MEMIDX_NO_CORPUS_CONFIG=1`) skips the corpus file.
"""
import fnmatch
import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime, timezone as _utc_tz
from pathlib import Path

from . import embed as _embed
from . import lang

CONFIG_NAME = "memidx.toml"
USER_CONFIG_NAME = "config.toml"

# Keys a config file inside the indexed repository may set. Everything else needs a trusted source:
# `embedder` imports code, `db`/`log_path`/`model_cache` choose write locations, `private_*` decide who sees
# what (a branch could switch `private_mode = "all"`), `server` opens ports, `vectors`/`model` load a model.
CORPUS_KEYS = frozenset({"languages", "timezone", "exclude", "default_zone", "zones", "vector_min_weight", "hints",
                         "renames", "memory_entries_prefix", "ranking"})
MAX_PATTERN_CHARS = 500          # limits for regexes from an untrusted file
MAX_PATTERNS = 500

# Environment variables that change where data is read from or written to (reported by update/search/serve).
LOCATION_ENV = ("MEMIDX_CONFIG", "MEMIDX_DB", "MEMIDX_LOG", "MEMIDX_MODEL_CACHE", "MEMIDX_EMBEDDER")


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
    source: str = ""                                  # last config file applied ("" = defaults)
    sources: tuple = ()                               # ((path, "user"|"corpus"|"explicit"), ...)
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
    vectors: bool = False                             # opt-in; needs `model` (or an `embedder` hook)
    model: str = ""                                   # model preset name, see embed.PRESETS
    model_cache: str = ""
    embedder: str = ""                                # "module:factory" hook (trusted sources only)
    threads: int = 0                                  # onnxruntime intra-op threads (0 = default)
    graph_workers: int = 2
    git_safe_directory: bool = False                  # pass `-c safe.directory=*` to git (trusted only)
    host: str = "127.0.0.1"
    port: int = 8800
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8801
    mcp_client: str = "mcp"
    interval: int = 600
    # how the configuration was built (for status, notices and the MCP child process)
    explicit_config: str = ""
    isolated: bool = False
    corpus_config: bool = True
    db_is_default: bool = False
    env_used: tuple = ()                              # MEMIDX_* names that changed the configuration
    notices: tuple = ()                               # one-line notices for stderr

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

    def location_env(self):
        """MEMIDX_* names in effect that change where data is read from or written to."""
        return tuple(n for n in LOCATION_ENV if n in self.env_used)


def today_in(tzname):
    if tzname:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(tzname)).date()
        except Exception:
            return datetime.now(_utc_tz.utc).date()   # unknown zone or no tz database: UTC
    return datetime.now().astimezone().date()


# ---------------------------------------------------------------- per-user locations
def _abs_env(env, name):
    v = (env or {}).get(name, "")
    return v if v and os.path.isabs(v) else ""


def user_config_path(env=None):
    env = os.environ if env is None else env
    if os.name == "nt":
        base = _abs_env(env, "APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = _abs_env(env, "XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "memidx" / USER_CONFIG_NAME


def user_cache_dir(env=None):
    env = os.environ if env is None else env
    if os.name == "nt":
        base = _abs_env(env, "LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        base = _abs_env(env, "XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "memidx"


def default_model_cache(env=None):
    env = os.environ if env is None else env
    base = env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "memidx", "models")


def default_db_path(corpus, env=None):
    """<user cache>/memidx/index/<basename>-<16 hex of sha256(resolved corpus path)>/index.db (outside the corpus)."""
    resolved = str(Path(corpus).resolve())
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(resolved).name) or "corpus"
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return str(user_cache_dir(env) / "index" / f"{name}-{digest}" / "index.db")


def legacy_db_path(corpus):
    return str(Path(corpus).resolve() / ".memidx" / "index.db")


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
             "vectors", "model", "model_cache", "embedder", "graph_workers", "git_safe_directory", "db", "server"}
_SERVER_KEYS = {"host", "port", "mcp_host", "mcp_port", "mcp_client", "interval"}


def find_config(explicit=None, corpus=None, env=None):
    """The most specific config file (explicit, then MEMIDX_CONFIG, then the corpus file). Kept for callers of
    0.1.0; load_config() layers all sources."""
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


def _safe_hints(items, where, notices):
    """Hints from an untrusted file: bad entries are skipped with a notice, never a crash."""
    out, skipped = [], []
    items = list(items) if isinstance(items, list) else []
    if len(items) > MAX_PATTERNS:
        notices.append(f"{where}: only the first {MAX_PATTERNS} of {len(items)} hints are used")
        items = items[:MAX_PATTERNS]
    for i, h in enumerate(items):
        try:
            pat, text = str(h["pattern"]), str(h["text"])
            if len(pat) > MAX_PATTERN_CHARS:
                raise ValueError(f"pattern longer than {MAX_PATTERN_CHARS} characters")
            re.compile(pat)
        except (TypeError, KeyError, ValueError, re.error, RecursionError, OverflowError) as e:
            skipped.append(f"#{i + 1} ({type(e).__name__}: {str(e)[:80]})")
            continue
        out.append((pat, text))
    if skipped:
        notices.append(f"{where}: skipped invalid hint(s): {', '.join(skipped)}")
    return tuple(out)


def _safe_renames(items, where, notices):
    out, skipped = [], []
    items = list(items) if isinstance(items, list) else []
    if len(items) > MAX_PATTERNS:
        notices.append(f"{where}: only the first {MAX_PATTERNS} of {len(items)} renames are used")
        items = items[:MAX_PATTERNS]
    for i, r in enumerate(items):
        try:
            r = dict(r)
            names = [str(r.get("old", ""))] + [str(v) for v in r.get("variants", [])]
            if len(names) > MAX_PATTERNS or any(len(n) > MAX_PATTERN_CHARS for n in names):
                raise ValueError("too many or too long names")
            r["variants"] = names[1:]
            rename_hints(r)
        except (TypeError, ValueError, AttributeError) as e:
            skipped.append(f"#{i + 1} ({type(e).__name__}: {str(e)[:80]})")
            continue
        out.append(r)
    if skipped:
        notices.append(f"{where}: skipped invalid rename(s): {', '.join(skipped)}")
    return tuple(out)


def _from_dict(data, base, cfg_dir, trusted=True, where="config", notices=None):
    notices = [] if notices is None else notices
    unknown = set(data) - _TOP_KEYS
    if unknown and trusted:
        raise ValueError(f"{where}: unknown config key(s): {sorted(unknown)}")
    kw = {}
    if "languages" in data:
        kw["languages"] = lang.check_languages(data["languages"])
    for k in ("timezone", "own_client_prefix", "private_mode", "memory_entries_prefix", "embedder", "model"):
        if k in data:
            kw[k] = str(data[k])
    if "vector_min_weight" in data:
        kw["vector_min_weight"] = float(data["vector_min_weight"])
    for k in ("vectors", "git_safe_directory"):
        if k in data:
            kw[k] = bool(data[k])
    if "graph_workers" in data:
        kw["graph_workers"] = int(data["graph_workers"])
    for k in ("exclude", "private_prefixes"):
        if k in data:
            kw[k] = tuple(str(x) for x in data[k] if str(x))
    if "zones" in data:
        kw["zones"] = tuple(_zone(z, f"{where}: zones") for z in data["zones"])
    if "default_zone" in data:
        d = data["default_zone"]
        kw["default_zone"] = ZoneRule(str(d.get("name", "default")), float(d.get("weight", 1.0)))
    if "hints" in data:
        if trusted:
            hs = []
            for h in data["hints"]:
                re.compile(h["pattern"])
                hs.append((str(h["pattern"]), str(h["text"])))
            kw["hints"] = tuple(hs)
        else:
            kw["hints"] = _safe_hints(data["hints"], where, notices)
    if "renames" in data:
        kw["renames"] = tuple(dict(r) for r in data["renames"]) if trusted else _safe_renames(data["renames"], where, notices)
    if "ranking" in data:
        r = dict(data["ranking"])
        bad = set(r) - set(Ranking.__dataclass_fields__)
        if bad:
            raise ValueError(f"{where}: unknown ranking key(s): {sorted(bad)}")
        kw["ranking"] = replace(base.ranking, **r)
    for k in ("log_path", "model_cache", "db"):
        if k in data and data[k]:
            p = Path(os.path.expanduser(str(data[k])))
            kw[k] = str(p if p.is_absolute() or cfg_dir is None else cfg_dir / p)
    if "server" in data:
        s = dict(data["server"])
        bad = set(s) - _SERVER_KEYS
        if bad:
            raise ValueError(f"{where}: unknown server key(s): {sorted(bad)}")
        for k in ("host", "mcp_host", "mcp_client"):
            if k in s:
                kw[k] = str(s[k])
        for k in ("port", "mcp_port", "interval"):
            if k in s:
                kw[k] = int(s[k])
    return replace(base, **kw)


def _read_toml(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: invalid TOML: {e}") from e
    except OSError as e:
        raise ValueError(f"cannot read config file {path}: {e.strerror or e}") from e


def _load_corpus_file(path, cfg, notices, user_path):
    """The untrusted corpus file: only CORPUS_KEYS are applied; one notice names the ignored keys."""
    data = _read_toml(path)
    ignored = sorted(set(data) - CORPUS_KEYS)
    allowed = {k: v for k, v in data.items() if k in CORPUS_KEYS}
    try:
        cfg = _from_dict(allowed, cfg, None, trusted=False, where=str(path), notices=notices)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise ValueError(f"{e} (in the repository's {CONFIG_NAME}; skip that file with --no-corpus-config)") from e
    if ignored:
        notices.append(f"ignored key(s) in {path}: {', '.join(ignored)}. A {CONFIG_NAME} inside the indexed "
                       f"repository is not trusted and may only set: {', '.join(sorted(CORPUS_KEYS))}. Set the other "
                       f"keys in the user config ({user_path}), with --config/MEMIDX_CONFIG, or with MEMIDX_* "
                       "variables and command-line flags.")
    return cfg


def _same_file(a, b):
    try:
        return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _flag(v):
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def apply_env(cfg, env=None):
    """MEMIDX_* environment variables override the files (names kept from the original service). The names
    that changed something are recorded in `env_used`."""
    env = os.environ if env is None else env
    kw, rk, used = {}, {}, []

    def take(name, nonempty=True):
        v = env.get(name) if nonempty else (env[name] if name in env else None)
        if (v if nonempty else v is not None):
            used.append(name)
            return v
        return None

    if (v := take("MEMIDX_LANGUAGES")) is not None:
        kw["languages"] = lang.check_languages(v.split(","))
    if (v := take("MEMIDX_TIMEZONE")) is not None:
        kw["timezone"] = v
    if env.get("MEMIDX_ZONES") and os.path.exists(env["MEMIDX_ZONES"]):
        used.append("MEMIDX_ZONES")
        with open(env["MEMIDX_ZONES"], encoding="utf-8") as f:
            kw["zones"] = tuple(ZoneRule(z[1], float(z[2]), prefix=z[0]) for z in json.load(f))
    if env.get("MEMIDX_TOMBSTONES") and os.path.exists(env["MEMIDX_TOMBSTONES"]):
        used.append("MEMIDX_TOMBSTONES")
        with open(env["MEMIDX_TOMBSTONES"], encoding="utf-8") as f:
            kw["hints"] = tuple((t[0], t[1]) for t in json.load(f))
        kw["renames"] = ()                       # the original replaced the whole hint list
    if (v := take("MEMIDX_EXCLUDE", nonempty=False)) is not None:
        kw["exclude"] = tuple(d for d in v.split(",") if d)
    if (v := take("MEMIDX_PRIVATE_PREFIXES", nonempty=False)) is not None:
        kw["private_prefixes"] = tuple(p for p in v.split(",") if p)
    if (v := take("MEMIDX_OWN_PREFIX")) is not None:
        kw["own_client_prefix"] = v
    if (v := take("MEMIDX_PRIVATE_MODE")) is not None:
        kw["private_mode"] = v
    for name in ("MEMIDX_MEMORY_PREFIX", "MEMIDX_GEDAECHTNIS_PREFIX"):
        if (v := take(name, nonempty=False)) is not None:
            kw["memory_entries_prefix"] = v
    if "MEMIDX_LOG" in env:
        kw["log_path"] = env["MEMIDX_LOG"]
        if env["MEMIDX_LOG"]:
            used.append("MEMIDX_LOG")
    if (v := take("MEMIDX_VECTORS", nonempty=False)) is not None:
        kw["vectors"] = _flag(v)
    if (v := take("MEMIDX_MODEL")) is not None:
        kw["model"] = v
    if (v := take("MEMIDX_MODEL_CACHE")) is not None:
        kw["model_cache"] = v
    if (v := take("MEMIDX_EMBEDDER")) is not None:
        kw["embedder"] = v
    if (v := take("MEMIDX_THREADS")) is not None:
        kw["threads"] = int(v) if v.strip().isdigit() else 0
    if (v := take("MEMIDX_GRAPH_WORKERS")) is not None:
        kw["graph_workers"] = int(v)
    if (v := take("MEMIDX_GIT_SAFE_DIRECTORY", nonempty=False)) is not None:
        kw["git_safe_directory"] = _flag(v)
    if (v := take("MEMIDX_DB")) is not None:
        kw["db"] = v
    if (v := take("MEMIDX_MCP_HOST")) is not None:
        kw["mcp_host"] = v
    if (v := take("MEMIDX_MCP_PORT")) is not None:
        kw["mcp_port"] = int(v)
    if (v := take("MEMIDX_MCP_CLIENT")) is not None:
        kw["mcp_client"] = v
    if (v := take("MEMIDX_INTERVAL")) is not None:
        kw["interval"] = int(v)
    if (v := take("MEMIDX_RECENCY", nonempty=False)) is not None:
        rk["recency"] = v == "1"
    if (v := take("MEMIDX_LINKS", nonempty=False)) is not None:
        rk["links"] = v != "0"
    if env.get("MEMIDX_LINKS_MAX"):
        try:
            rk["links_max"] = float(env["MEMIDX_LINKS_MAX"])
            used.append("MEMIDX_LINKS_MAX")
        except ValueError:
            pass
    for name, key, typ in (("MEMIDX_RRF_BM", "rrf_bm25", float), ("MEMIDX_RRF_VEC", "rrf_vector", float),
                           ("MEMIDX_RRF_DOC", "rrf_doc", float), ("MEMIDX_RRF_DOCTOP", "rrf_doc_top", int)):
        if (v := take(name)) is not None:
            rk[key] = typ(v)
    if rk:
        kw["ranking"] = replace(cfg.ranking, **rk)
    kw["env_used"] = tuple(dict.fromkeys(cfg.env_used + tuple(used)))
    return replace(cfg, **kw)


def strip_memidx_env(env):
    return {k: v for k, v in env.items() if not k.startswith("MEMIDX_")}


def load_config(path=None, corpus=None, db=None, env=None, *, isolated=False, corpus_config=None,
                trust_repo_owner=False):
    """Build the effective configuration. `path`, `corpus`, `db` and `trust_repo_owner` are explicit values
    (CLI flags) and win over everything else."""
    env = dict(os.environ if env is None else env)
    if isolated:
        env = strip_memidx_env(env)
    used = []
    if not corpus and env.get("MEMIDX_CORPUS"):
        used.append("MEMIDX_CORPUS")
    corpus = corpus or env.get("MEMIDX_CORPUS") or os.getcwd()
    corpus_abs = Path(corpus).resolve()
    if corpus_config is None:
        corpus_config = not _flag(env.get("MEMIDX_NO_CORPUS_CONFIG", ""))
        if not corpus_config:
            used.append("MEMIDX_NO_CORPUS_CONFIG")
    explicit = None
    if path:
        explicit = Path(path)
    elif env.get("MEMIDX_CONFIG"):
        explicit = Path(env["MEMIDX_CONFIG"])
        used.append("MEMIDX_CONFIG")
    if explicit is not None and not explicit.is_file():
        raise ValueError(f"config file not found: {explicit}")

    cfg, sources, notices = Config(env_used=tuple(used)), [], []
    user_path = user_config_path(env)
    if not isolated and user_path.is_file() and not _same_file(user_path, explicit):
        cfg = _from_dict(_read_toml(user_path), cfg, user_path.parent.resolve(), where=str(user_path))
        sources.append((str(user_path), "user"))
    corpus_file = corpus_abs / CONFIG_NAME
    if corpus_config and corpus_file.is_file() and not _same_file(corpus_file, explicit) \
            and not _same_file(corpus_file, user_path):
        cfg = _load_corpus_file(corpus_file, cfg, notices, user_path)
        sources.append((str(corpus_file), "corpus"))
    if explicit is not None:
        cfg = _from_dict(_read_toml(explicit), cfg, explicit.parent.resolve(), where=str(explicit))
        sources.append((str(explicit), "explicit"))
    cfg = apply_env(cfg, env)

    kw = {"corpus": str(corpus_abs), "sources": tuple(sources), "source": sources[-1][0] if sources else "",
          "explicit_config": str(explicit) if explicit is not None else "", "isolated": bool(isolated),
          "corpus_config": bool(corpus_config)}
    env_used = list(cfg.env_used)
    if db:
        kw["db"] = str(db)
        if "MEMIDX_DB" in env_used:
            env_used.remove("MEMIDX_DB")          # the flag wins: the variable has no effect
    elif not cfg.db:
        kw["db"] = default_db_path(corpus_abs, env)
        kw["db_is_default"] = True
        legacy = legacy_db_path(corpus_abs)
        if os.path.isfile(legacy):
            notices.append(f"{legacy} is no longer used (the index now lives at {kw['db']}); "
                           "you can delete the .memidx directory in the repository")
    if trust_repo_owner:
        kw["git_safe_directory"] = True
    if not cfg.model_cache:
        kw["model_cache"] = default_model_cache(env)
    model = cfg.model.strip()
    if model:
        _embed.get_preset(model)        # ValueError (exit 2) for an unknown or removed preset
    kw["model"] = model
    kw["env_used"] = tuple(env_used)
    kw["notices"] = tuple(notices)
    return replace(cfg, **kw)


def check_vectors(cfg):
    """vectors = true needs an explicit model preset or an embedder hook: never guess a model."""
    if cfg.vectors and not cfg.model and not cfg.embedder and _embed.current_embedder() is None:
        raise ValueError("vectors = true but no model is configured. Choose a preset with `model = \"<preset>\"` in "
                         "the user config, --config file or MEMIDX_MODEL (see `memidx model list`), or set "
                         "vectors = false. Full-text search needs no model.")
    return cfg


def is_inside(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


_EXAMPLE_INTRO = '''# memidx configuration. All keys are optional; this file shows the defaults or commented examples.
'''

_EXAMPLE_CORPUS_HEAD = '''# This file lives inside the indexed repository. memidx does NOT trust it: only the keys below
# (languages, timezone, exclude, default_zone, zones, vector_min_weight, hints, renames,
# memory_entries_prefix, ranking) are read from it. Everything else (db, log_path, vectors, model, model_cache,
# embedder, private_*, server, git_safe_directory, ...) belongs in the user config:
#   {user_config}
# Create it with `memidx init --user`. `memidx --no-corpus-config ...` skips this file completely.
'''

_EXAMPLE_USER_HEAD = '''# Trusted file (user config, --config or MEMIDX_CONFIG): every key may be set here.
# Layering, lowest first: defaults < user config < <corpus>/memidx.toml (limited keys) < --config/MEMIDX_CONFIG
# < MEMIDX_* environment variables < command-line flags.
'''

# top-level keys first: in TOML every key after a [table] header belongs to that table
_EXAMPLE_TOP = '''
# Languages for full-text normalization: "de", "en" or both. Changing this rebuilds the full-text index on
# the next `memidx update`.
languages = ["de", "en"]

# Time zone for "today" (validity dates of memory entries). "" = system local time.
timezone = ""

# Directory prefixes that are not indexed.
exclude = []

# Zones with a weight below this value are searched by full text only (no vectors).
vector_min_weight = 0.7

# Files under this prefix are memory entries with their own validity rules ("" = off).
memory_entries_prefix = ""
'''

_EXAMPLE_USER_TOP = '''
# Where the index lives. Relative paths are relative to this file.
# Default: <user cache>/memidx/index/<repo name>-<hash>/index.db (outside the repository)
# db = "/path/to/index.db"

# Paths under these prefixes are only visible to clients whose self-declared client id starts with
# own_client_prefix. This is a convention against accidental disclosure, NOT access control.
private_prefixes = []
own_client_prefix = "own:"
private_mode = "own"          # "all" = everyone sees private paths

# Append one JSON line per query from an identified client to this file ("" = off).
log_path = ""

# Vectors are opt-in. They need `pip install memidx[vectors]`, an explicit model preset and
# `memidx model download <preset>`. See `memidx model list` for presets and their licenses.
vectors = false
# model = "arctic-embed-m-v2.0"   # Apache-2.0; `memidx model list` shows all presets and their licenses
# model_cache = "~/.cache/memidx/models"
# embedder = "package.module:factory"   # your own embedder; imports and runs that code

# git refuses repositories owned by another user ("dubious ownership"). true = memidx passes
# `-c safe.directory=*` to its own git calls (for example a mounted repository in a container).
git_safe_directory = false
'''

_EXAMPLE_TABLES = '''
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

[ranking]
rrf_bm25 = 1.0
rrf_vector = 1.0
rrf_doc = 0.3
rrf_doc_top = 20
stale_factor = 0.6
links = true
links_max = 0.10
recency = false
'''

_EXAMPLE_SERVER = '''
[server]
host = "127.0.0.1"
port = 8800
mcp_host = "127.0.0.1"
mcp_port = 8801
mcp_client = "mcp"
interval = 600
'''


def example_corpus(user_config=None):
    """Commented example for <corpus>/memidx.toml: only the keys that file may set."""
    head = _EXAMPLE_CORPUS_HEAD.replace("{user_config}", str(user_config or user_config_path()))
    return _EXAMPLE_INTRO + head + _EXAMPLE_TOP + _EXAMPLE_TABLES


EXAMPLE_CORPUS = _EXAMPLE_INTRO + _EXAMPLE_CORPUS_HEAD.replace(
    "{user_config}", "$XDG_CONFIG_HOME/memidx/config.toml, ~/.config/memidx/config.toml or %APPDATA%\\memidx\\config.toml"
) + _EXAMPLE_TOP + _EXAMPLE_TABLES
EXAMPLE = _EXAMPLE_INTRO + _EXAMPLE_USER_HEAD + _EXAMPLE_TOP + _EXAMPLE_USER_TOP + _EXAMPLE_TABLES + _EXAMPLE_SERVER
