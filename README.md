# memidx

memidx indexes the Markdown files of a git repository as **sections** and answers a query with only the matching
sections (path, heading path, line range, text), limited by a token budget. It is built for agents and people who
need the relevant part of a large notes or docs repository, not whole files.

- Full-text search with SQLite FTS5 over normalized text (German and/or English folding, stopwords, light stemming).
- Optional, opt-in local vector search (ONNX on CPU), fused with full text by reciprocal rank fusion. memidx ships
  no model; see [Vectors](#vectors-opt-in).
- Zones: path-based groups with score weights (for example, archives count less than curated notes).
- Incremental updates: one content hash per file; only changed files are re-indexed, vectors of unchanged
  sections are reused. The server can re-index on an interval.
- Extras: hints for renamed or retired names, wikilink edges with a small ranking bonus, stale/superseded marking,
  memory entries with validity dates, private path prefixes, query log, secret scrubbing.
- Interfaces: CLI, HTTP JSON API with a small read UI, MCP server (stdio or streamable HTTP).

Files in git stay the source of truth. The index is derived and can be deleted and rebuilt at any time.

## Quick start

```bash
# from a checkout of this repository (not published on PyPI)
pip install .                      # full text only, no third-party dependencies (tzdata on Windows)
pip install ".[mcp]"               # + MCP server

cd /path/to/markdown-repo
memidx init                        # optional: writes a commented memidx.toml (repository-safe keys only)
memidx update                      # builds the index in your user cache directory (see below)
memidx search "how are backups restored"
memidx search --list backup        # one line per hit, then:
memidx get notes/backup.md:12-20   # read a line range
memidx status                      # index statistics, config files and MEMIDX_* variables in effect
memidx serve                       # http://127.0.0.1:8800 (API + UI), re-indexes every 600 s
```

**Where the index lives.** By default the index is stored outside the repository, in
`<user cache>/memidx/index/<repo name>-<16 hex digits>/index.db`, where the user cache is `$XDG_CACHE_HOME` or
`~/.cache` (Windows: `%LOCALAPPDATA%`) and the digits are the start of the SHA-256 of the resolved repository path.
On POSIX systems memidx creates these directories with mode 0700 and the database, its `-wal`/`-shm` files and the
query log with mode 0600. On Windows these files inherit the ACL of your user profile directory. Set `db` in a
trusted config, `MEMIDX_DB` or `--db` to choose another place. A `<repo>/.memidx/index.db` from memidx 0.1.0 is no
longer used; memidx prints one notice and leaves it alone, and you can delete it.

**Exit codes.**

| Code | Meaning |
|---|---|
| 0 | Success; `search` found at least one hit. |
| 1 | `search` ran correctly but found nothing (`reason: no_match`); `get`: unknown path. |
| 2 | Usage or configuration error, also an unknown zone (the message lists the zones in the index) and a query that is empty after normalization (`reason: empty_query`, the message shows the dropped words). |
| 3 | No usable index (missing file, empty index, missing schema, or the index was built with other normalization rules and needs `memidx update`), `git ls-files` failed in a git work tree, or the remote server is unreachable. |

Only `update` (and the auto-update of `serve`) creates an index. `search`, `status`, `get`, `mcp` and every HTTP
request open the database read-only and never create one.

### Vectors (opt-in)

memidx ships no embedding model, and full-text search needs none. Vectors are off by default (`vectors = false`).
To use them you choose a model preset explicitly:

```bash
pip install ".[vectors]"                    # numpy, onnxruntime, tokenizers, huggingface_hub
memidx model list                           # presets with their license and whether it is OSI-approved
memidx model download embeddinggemma-300m --accept-license
# then, in the user config or a --config file:  vectors = true  and  model = "embeddinggemma-300m"
```

**The only preset today, `embeddinggemma-300m`, is distributed by Google under the Gemma Terms of Use, which is
not an OSI-approved open-source license.** `memidx model download` refuses to fetch a preset whose license is not
OSI-approved unless you pass `--accept-license`; read the terms first. memidx never downloads a model on its own.
With `vectors = true` but no `model` (and no `embedder` hook), `update`, `search`, `serve` and `mcp` stop with exit
code 2 instead of guessing a model. If the extra or the model files are missing, memidx falls back to full text and
prints one notice line.

### Docker

The image was built and smoke-tested on Linux (amd64) with full-text search; the vector path inside the container
is not tested. It installs the `vectors` and `mcp` extras, runs as a non-root user, and uses `memidx` as PID 1, which handles SIGTERM. Vectors stay off unless
you set `MEMIDX_VECTORS=1` and `MEMIDX_MODEL=<preset>`.

```bash
docker build -t memidx .
docker run --rm -e HF_HUB_OFFLINE=0 -v memidx-models:/models memidx \
  model download embeddinggemma-300m --accept-license          # Gemma Terms of Use apply
docker run -d --name memidx -p 127.0.0.1:8800:8800 -p 127.0.0.1:8801:8801 \
  -v /path/to/repo:/corpus:ro -v memidx-data:/data -v memidx-models:/models memidx
```

Volumes: `/corpus` (repository, read-only; its `memidx.toml` is read with the restricted key set), `/data` (index
and query log), `/models` (model cache). The image sets `MEMIDX_GIT_SAFE_DIRECTORY=1` because the mounted
repository belongs to a host user, not to the container user, and git would otherwise refuse it. Inside the
container the servers bind `0.0.0.0`; publish the ports on `127.0.0.1` unless you add authentication in front.

## A pointer, not proof

A memidx hit points you to text that exists. The absence of a hit does **not** prove that the text is absent from
the repository. Each of these can hide existing text:

- normalization (case and accent folding, `ß`→`ss`) and tokenization,
- stopwords, which are dropped from both the query and the index,
- stemming, which is a heuristic and can map words apart or together,
- the per-file cap (at most 2 sections per file) and `k`,
- the token budget, which skips hits that do not fit,
- private filtering (`private_excluded: true` and the line "private paths were not searched"),
- untracked files in a git checkout, `exclude`, and invalid memory entries.

Every empty result carries a `reason` and a `message`. To check that something is absent, use `git grep` or `grep`.

## Security model

memidx is meant to be safe to run on a checkout you do not fully trust (for example a colleague's branch or a
cloned repository). The details and how to report a problem are in [SECURITY.md](SECURITY.md).

- **Trusted:** command-line flags, `MEMIDX_*` environment variables, the user config file and a file passed with
  `--config` or `MEMIDX_CONFIG`. These may set every key.
- **Untrusted:** the repository content, including `<repo>/memidx.toml`. That file may only set `languages`,
  `timezone`, `exclude`, `default_zone`, `zones`, `vector_min_weight`, `hints`, `renames`,
  `memory_entries_prefix` and `ranking`. Every other key (`embedder`, `db`, `log_path`, `model_cache`, `vectors`,
  `model`, `server`, `private_*`, `git_safe_directory`, `graph_workers`, unknown keys) is ignored with one notice.
  Hint regexes from it are length- and count-limited; an invalid one is skipped, never a crash.
  `--no-corpus-config` or `MEMIDX_NO_CORPUS_CONFIG=1` skips the file completely.
- **Environment:** `--isolated` (`-I`) ignores all `MEMIDX_*` variables and the user config for one invocation.
  `memidx status` lists which `MEMIDX_*` variables are set and which are in effect (names only). When
  `MEMIDX_CONFIG`, `MEMIDX_DB`, `MEMIDX_LOG`, `MEMIDX_MODEL_CACHE` or `MEMIDX_EMBEDDER` is in effect, `update`,
  `search` and `serve` print one notice that names them. `search` only goes over HTTP with an explicit `--url`.
- **git:** memidx calls only `git ls-files` with `-c core.fsmonitor=false -c core.untrackedCache=false` and
  `GIT_OPTIONAL_LOCKS=0`, so a repository-local git config cannot start programs through it. git's ownership check
  stays on; for a repository owned by another user set `git_safe_directory = true` (trusted config),
  `MEMIDX_GIT_SAFE_DIRECTORY=1` or `--trust-repo-owner`.
- **No authentication** on the HTTP and MCP servers. The client id is self-declared. Keep them on `127.0.0.1`
  (the default) or put an authenticating proxy in front.
- **Start it as `memidx`, not `python -m memidx`, in a checkout you do not trust.** With `-m`, Python puts the
  current directory first on its module search path and would import a `json.py` from the repository. If you need
  `-m`, use `python -P -m memidx` or set `PYTHONSAFEPATH=1`.

## Configuration

Configuration comes from TOML files (read with `tomllib`), `MEMIDX_*` environment variables and flags. Layering,
lowest first:

1. built-in defaults,
2. the user config: `$XDG_CONFIG_HOME/memidx/config.toml` or `~/.config/memidx/config.toml`
   (Windows: `%APPDATA%\memidx\config.toml`); create it with `memidx init --user`,
3. `<corpus>/memidx.toml`, only the repository-safe keys (see [Security model](#security-model)),
4. an explicit file: `--config PATH`, else `MEMIDX_CONFIG`,
5. `MEMIDX_*` environment variables,
6. command-line flags.

Relative paths in a file are relative to that file. In trusted files unknown keys are an error. `memidx init`
writes a commented example: inside the repository with only the repository-safe keys, with `--user` the full
example to the user config path.

| Key | Default | Repo file | Meaning |
|---|---|---|---|
| `languages` | `["de", "en"]` | yes | Normalization rules. Changing it rebuilds the full-text table on the next `update`. |
| `timezone` | `""` (system local) | yes | Time zone for "today" in validity checks. Unknown zone: UTC. |
| `exclude` | `[]` | yes | Directory prefixes that are not indexed. |
| `[default_zone]` | `name="default"`, `weight=1.0` | yes | Zone for files that match no rule. |
| `[[zones]]` | none | yes | `name`, `weight` and either `prefix` (ends with `/` = directory, else exact path) or `basename` (file-name glob). Basename rules are checked first, then prefix rules in order; first match wins. |
| `vector_min_weight` | `0.7` | yes | Zones below this weight get no vectors (full text only). |
| `[[hints]]` | none | yes | `pattern` (regex on the case-folded query), `text`. |
| `[[renames]]` | none | yes | `old`, `new`, `since`, optional `variants`, optional `text`. Generates the hint "`old` is called `new` since `since`." for each spelling. |
| `memory_entries_prefix` | `""` (off) | yes | Files under this prefix are memory entries (see below). |
| `[ranking]` | see below | yes | `rrf_bm25=1.0`, `rrf_vector=1.0`, `rrf_doc=0.3`, `rrf_doc_top=20`, `stale_factor=0.6`, `links=true`, `links_max=0.10`, `recency=false`. |
| `db` | user cache, see above | no | Index file. |
| `private_prefixes` | `[]` | no | Path prefixes visible only to "own" clients (see below). |
| `own_client_prefix` | `"own:"` | no | Client ids starting with this see private paths. |
| `private_mode` | `"own"` | no | `"all"` = everyone sees private paths. |
| `log_path` | `""` (off) | no | JSON-lines query log for requests with a client id. |
| `vectors` | `false` | no | `true` = hybrid search; needs `model` or `embedder`. |
| `model` | `""` | no | Model preset name (`memidx model list`). No model is implied. |
| `model_cache` | `~/.cache/memidx/models` | no | Model directory. |
| `embedder` | `""` | no | `"module:factory"` to plug in your own embedder (imports that code; see below). |
| `graph_workers` | `2` | no | Threads for `/graph`. |
| `git_safe_directory` | `false` | no | `true` = pass `-c safe.directory=*` to git (repository owned by another user). |
| `[server]` | | no | `host="127.0.0.1"`, `port=8800`, `mcp_host="127.0.0.1"`, `mcp_port=8801`, `mcp_client="mcp"`, `interval=600`. |

Environment overrides: `MEMIDX_CORPUS`, `MEMIDX_CONFIG`, `MEMIDX_NO_CORPUS_CONFIG` (`1` = skip the repository
file), `MEMIDX_DB`, `MEMIDX_LANGUAGES` (comma list), `MEMIDX_TIMEZONE`, `MEMIDX_EXCLUDE`, `MEMIDX_ZONES` (JSON file
`[["prefix","zone",weight],...]`, replaces all zone rules), `MEMIDX_TOMBSTONES` (JSON file `[["regex","text"],...]`,
replaces hints and renames), `MEMIDX_PRIVATE_PREFIXES`, `MEMIDX_OWN_PREFIX`, `MEMIDX_PRIVATE_MODE`,
`MEMIDX_MEMORY_PREFIX` (alias `MEMIDX_GEDAECHTNIS_PREFIX`), `MEMIDX_LOG`, `MEMIDX_VECTORS` (`0`/`1`),
`MEMIDX_MODEL`, `MEMIDX_MODEL_CACHE`, `MEMIDX_EMBEDDER`, `MEMIDX_THREADS` (onnxruntime threads),
`MEMIDX_GRAPH_WORKERS`, `MEMIDX_GIT_SAFE_DIRECTORY` (`1` = on), `MEMIDX_RRF_BM`, `MEMIDX_RRF_VEC`,
`MEMIDX_RRF_DOC`, `MEMIDX_RRF_DOCTOP`, `MEMIDX_LINKS` (`0` = off), `MEMIDX_LINKS_MAX`, `MEMIDX_RECENCY`
(`1` = on), `MEMIDX_MCP_HOST`, `MEMIDX_MCP_PORT`, `MEMIDX_MCP_CLIENT`, `MEMIDX_MCP_TRANSPORT`, `MEMIDX_INTERVAL`
and `MEMIDX_CLIENT` (CLI `search`/`get`). `MEMIDX_URL` is no longer read: remote search needs `--url`.

### Languages

German rules: fold case, `ß`→`ss`, `ä/ö/ü/é/è/á/à`→base letter; German stopwords (plus a few English ones); strip
the first matching suffix of `ungen ung heiten heit keiten keit schaft ern ers en er es em e s n` if at least 4
letters remain (words of 4 letters or fewer and numbers stay). English rules: English stopwords; strip one of
`sses→ss`, `ies→i`, `-es` after `ch/sh/x/z`, a final `s` (not `ss/us/is`), or `-ing`/`-ed` (stem keeps a vowel and
3+ letters, a doubled final consonant is undoubled); then a final `y` after a consonant becomes `i`.

**Combined rule (both active):** stopwords are the union. Stemmers run in the fixed order German, then English; the
first one that changes the word wins. German stems are therefore never changed by the English rules; English only
applies to words the German rules leave alone (for example `indexing`, `query`). The index stores the language set
and a normalization version; when either changes, `update` rebuilds the full-text table from the stored sections.

### Frontmatter

`name` (or `title`) and `description` are used for titles, wikilink resolution and the description field.
Validity keys accept the original German names and English aliases:

| German | English | Meaning |
|---|---|---|
| `ersetzt_durch` | `superseded_by` | Successor; the document is stale. |
| `gueltig_bis` | `valid_until` | `YYYY-MM-DD`; stale after this day. |
| `gueltig_ab` | `valid_from` | Memory entries: start date. |
| `bestaetigt_am` | `confirmed_on` | Memory entries: confirmation date. |
| `art` | `kind` | Memory entries: `fakt`/`fact` or `praeferenz`/`preference`. |

`status: superseded` also marks a document stale. Stale hits are multiplied by `stale_factor` (0.6) and marked
`[STALE, superseded by ...]`; they are not hidden.

**Memory entries** (only under `memory_entries_prefix`): `status` must be `bestaetigt`/`confirmed`,
`widerrufen`/`revoked` or `ersetzt`/`superseded`; `valid_until` is a date or `bis_widerruf`/`until_revoked`.
Reasons are checked in this order: `format`, `revoked`, `superseded`, `not_yet_valid`, `expired`, else `valid`.
Invalid entries are hidden from search and related results unless `all=1` (CLI `--all`). Hits carry the JSON field
`memory` and the marker `[MEMORY Fact, valid until revoked, confirmed 2026-01-01]`.

### Client identity and private paths

Clients declare an id (`client=` query parameter, `--client`, `MEMIDX_CLIENT`). If it starts with
`own_client_prefix`, the client sees paths under `private_prefixes`; otherwise private files behave as if they did
not exist (search, `/doc`, `/related`, `/list`, `/graph`). **This is self-declared, not access control.** Anyone who
can reach the port can send any id. It protects against accidental disclosure to agents that use the default id.
The web UI and requests without an id never see private paths. The MCP server uses one fixed id (`mcp_client`).
When private prefixes are configured and the client does not see them, a search result carries
`"private_excluded": true` and the CLI prints "private paths were not searched"; whether anything matched there
is not revealed. Private settings can only come from trusted sources, so a branch cannot switch them off.
memidx has no authentication; keep it on `127.0.0.1` (the default) or put an authenticating proxy in front.

### Custom embedder

Any object with `embed_documents(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]` works.
Set it in Python with `memidx.embed.set_embedder(obj)` or in a trusted config (`embedder = "pkg.module:factory"`,
called with the config or without arguments; this imports and runs that code, so the repository file cannot set
it). An optional `name` attribute is stored in the index; when it changes, `update` discards the old vectors. The
tests use this hook with a small deterministic fake embedder.

## How ranking works

1. The query is normalized like the index. FTS5 returns up to 200 sections, ordered by BM25 with weight 5 on the
   heading field (heading path, file name, description) and 1 on the body.
2. With vectors: the 100 nearest section vectors and the `rrf_doc_top` (20) nearest document vectors (title,
   description and the first 800 characters) are added.
3. Reciprocal rank fusion: score = Σ weight / (60 + rank) over the lists (`rrf_bm25`, `rrf_vector`, `rrf_doc`;
   a document-vector rank counts for all sections of that document).
4. The fused score is multiplied by 1000, the zone weight, `stale_factor` for stale documents and the optional
   recency factor (up to +15 %, half-life 180 days, off by default).
5. Wikilink bonus: among the best 10 documents, a document linked (either direction) to 2 or more of them gets
   up to `+links_max` (10 %), scaled by `min(n, 4) / 4`.
6. Hits are taken in score order, at most 2 sections per file, until `k` hits or the token budget (4 characters ≈
   1 token) is used. Long sections are cut to the best-matching lines (1400 characters).

`related(path)` returns wikilink neighbours first, then a full-text search for the document's title and
description. `/graph` returns `/list`-ordered nodes and edges from `related` inside the node set, cached per index
state.

## Reference

### CLI

```
memidx [--config F] [--corpus DIR] [--db F] [--no-corpus-config] [-I | --isolated] [--trust-repo-owner] COMMAND
  init [--force] [--user]                         write a commented config (repository file, or --user: user config)
  update [--no-embed]                             index (incremental); embeds when vectors are enabled
  search QUERY... [-k N] [--budget T] [--zone Z] [--list | --json] [--all] [--client ID] [--url URL]
  get PATH[:FROM-TO] [--client ID]
  status
  serve [--host H] [--port P] [--interval S] [--no-auto-update] [--no-embed] [--mcp]
  mcp [--transport stdio|streamable-http] [--host H] [--port P]
  model list
  model download PRESET [--accept-license] [--cache-dir DIR]
```

The global flags work before or after the command. `search --url http://127.0.0.1:8800` queries a running server
instead of the local file (only with this explicit flag). `serve` indexes once at start, then every `interval`
seconds (0 = only at start); `--mcp` also starts the MCP server as a child process. `status` prints JSON with the
index statistics, the config files and their trust level, and the `MEMIDX_*` variable names that are set and in
effect; it exits with code 3 when the index is not usable.

### HTTP (default `127.0.0.1:8800`)

| Endpoint | Parameters | Result |
|---|---|---|
| `GET /search` | `q`, `k`=3, `budget`=800, `zone`, `mode`=`json`/`text`/`list`, `all`=1, `client` (default `http`) | `{"query", "hints", "hits": [{"path", "section", "lines", "zone", "score", "text", "stale"?, "superseded_by"?, "memory"?}], "tokens", "reason"?, "message"?, "private_excluded"?}` |
| `GET /doc` | `path`, `from`=1, `to`=0 (end), `client` | text, 404 for unknown or private paths |
| `GET /related` | `path`, `all`, `client` | list of hits (wikilink neighbours carry `"via": "link"`) |
| `GET /list` | `zone`, `prefix`, `limit`=200 (max 1000), `client` | `{"zones": {...}, "files": [{"path", "zone", "title", "descr", "sections"}]}` |
| `GET /graph` | `limit`=300 (max 1000), `zone`, `score`=1, `client` | `{"nodes": [{"path", "zone", "title", "mtime", "hash"}], "edges": [[a, b(, score)]], "n", "token", "computed_s"}` |
| `GET /status` | | `{"files", "sections", "vectors", "zones", "normalization", "embedder"}` |
| `GET /` | `q` | HTML read UI |

An empty `/search` result carries `reason` and `message`. Status codes: 200 for hits and for `no_match`, 400 for
`empty_query`, `unknown_zone` and bad numbers, 503 with `{"reason": "no_index", "detail", "message"}` on every
endpoint when there is no usable index. Each request opens its own read-only database connection.

### MCP

`memidx mcp` (stdio, default) or `memidx mcp --transport streamable-http` (`127.0.0.1:8801`). Tools:
`memory_search(query, k=3, budget_tokens=800, zone=None)`, `memory_get(path, from_line=1, to_line=0)`,
`memory_related(path, k=5)`, `memory_status()`. Works with `mcp` 1.x and 2.x. An empty `memory_search` answer
contains the reason (for example `(no hits) [no_match] ...`) and, when applicable, "(private paths were not
searched)". Without a usable index every tool answers `(no usable index) [no_index] ...`.

## Limits

- No authentication or authorization. The client id is self-declared.
- A hit is a pointer, not proof; see [A pointer, not proof](#a-pointer-not-proof).
- Stemming is a heuristic (German suffix stripping, light English rules). There is no synonym handling except
  what the vectors catch.
- Ranking constants were tuned and measured on one private German corpus. Those benchmark numbers and the test
  set are not included and do not transfer to other corpora.
- Only git-tracked `*.md` files are indexed in a git checkout (otherwise all `*.md` files in the tree).
  Frontmatter parsing is a simple `key: value` reader, not YAML.
- The first vector build is slow on a CPU (seconds per section); later updates only embed changed sections.
- One process should write the index at a time.

## Development

```bash
python -m venv .venv && .venv/bin/python -m pip install -e ".[dev,vectors,mcp]"
.venv/bin/python -m pytest -q
```

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see `LICENSE`. Third-party notices: `THIRD_PARTY_NOTICES.md`. memidx ships no embedding model. The only model
preset today (`embeddinggemma-300m`) is not part of this repository and is distributed by Google under the Gemma
Terms of Use, which is not an OSI-approved license.
