# memidx

memidx indexes the Markdown files of a git repository as **sections** and answers a query with only the matching
sections (path, heading path, line range, text), limited by a token budget. It is built for agents and people who
need the relevant part of a large notes or docs repository, not whole files.

- Full-text search with SQLite FTS5 over normalized text (German and/or English folding, stopwords, light stemming).
- Optional local vector search (EmbeddingGemma-300m, ONNX on CPU), fused with full text by reciprocal rank fusion.
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
pip install ".[vectors]"           # + local embeddings (numpy, onnxruntime, tokenizers, huggingface_hub)
pip install ".[mcp]"               # + MCP server

cd /path/to/markdown-repo
memidx init                        # optional: writes a commented memidx.toml
memidx update                      # builds .memidx/index.db
memidx search "how are backups restored"
memidx search --list backup        # one line per hit, then:
memidx get notes/backup.md:12-20   # read a line range
memidx status
memidx serve                       # http://127.0.0.1:8800 (API + UI), re-indexes every 600 s
```

Vectors (optional): install the `vectors` extra, then run `memidx model download`. This downloads about 218 MB
from Hugging Face into the model cache (`~/.cache/memidx/models` by default). **The model is distributed under the
Gemma Terms of Use, which you must accept yourself.** memidx never downloads the model on its own. Without the
extra or the model files, everything works with full text only and memidx prints one notice line.

The index lives at `<corpus>/.memidx/index.db` unless you set `db` in the config, `MEMIDX_DB` or `--db`. Add
`.memidx/` to the repository's `.gitignore`.

### Docker (untested)

The Dockerfile has not been built or run by the author of this package version. It installs the `vectors` and
`mcp` extras, runs as a non-root user, and uses `memidx` as PID 1, which handles SIGTERM.

```bash
docker build -t memidx .
docker run --rm -e HF_HUB_OFFLINE=0 -v memidx-models:/models memidx model download   # Gemma terms apply
docker run -d --name memidx -p 127.0.0.1:8800:8800 -p 127.0.0.1:8801:8801 \
  -v /path/to/repo:/corpus:ro -v memidx-data:/data -v memidx-models:/models memidx
```

Volumes: `/corpus` (repository, read-only; `memidx.toml` in it is read automatically), `/data` (index and query
log), `/models` (model cache). Inside the container the servers bind `0.0.0.0`; publish the ports on
`127.0.0.1` unless you add authentication in front.

## Configuration

`memidx.toml` is read with `tomllib`. Lookup: `--config`, then `MEMIDX_CONFIG`, then `<corpus>/memidx.toml`, then
built-in defaults. Relative paths in the file are relative to the file. `MEMIDX_*` environment variables override
the file; command-line flags override both. Unknown keys are an error.

| Key | Default | Meaning |
|---|---|---|
| `languages` | `["de", "en"]` | Normalization rules. Changing it rebuilds the full-text table on the next `update`. |
| `timezone` | `""` (system local) | Time zone for "today" in validity checks. Unknown zone: UTC. |
| `db` | `<corpus>/.memidx/index.db` | Index file. |
| `exclude` | `[]` | Directory prefixes that are not indexed. |
| `[default_zone]` | `name="default"`, `weight=1.0` | Zone for files that match no rule. |
| `[[zones]]` | none | `name`, `weight` and either `prefix` (ends with `/` = directory, else exact path) or `basename` (file-name glob). Basename rules are checked first, then prefix rules in order; first match wins. |
| `vector_min_weight` | `0.7` | Zones below this weight get no vectors (full text only). |
| `[[hints]]` | none | `pattern` (regex on the case-folded query), `text`. |
| `[[renames]]` | none | `old`, `new`, `since`, optional `variants`, optional `text`. Generates the hint "`old` is called `new` since `since`." for each spelling. |
| `private_prefixes` | `[]` | Path prefixes visible only to "own" clients (see below). |
| `own_client_prefix` | `"own:"` | Client ids starting with this see private paths. |
| `private_mode` | `"own"` | `"all"` = everyone sees private paths. |
| `memory_entries_prefix` | `""` (off) | Files under this prefix are memory entries (see below). |
| `log_path` | `""` (off) | JSON-lines query log for requests with a client id. |
| `vectors` | `true` | `false` = full text only. |
| `model_cache` | `~/.cache/memidx/models` | Model directory. |
| `embedder` | `""` | `"module:factory"` to plug in your own embedder (see below). |
| `graph_workers` | `2` | Threads for `/graph`. |
| `[ranking]` | see below | `rrf_bm25=1.0`, `rrf_vector=1.0`, `rrf_doc=0.3`, `rrf_doc_top=20`, `stale_factor=0.6`, `links=true`, `links_max=0.10`, `recency=false`. |
| `[server]` | | `host="127.0.0.1"`, `port=8800`, `mcp_host="127.0.0.1"`, `mcp_port=8801`, `mcp_client="mcp"`, `interval=600`. |

Environment overrides: `MEMIDX_CORPUS`, `MEMIDX_DB`, `MEMIDX_LANGUAGES` (comma list), `MEMIDX_TIMEZONE`,
`MEMIDX_EXCLUDE`, `MEMIDX_ZONES` (JSON file `[["prefix","zone",weight],...]`, replaces all zone rules),
`MEMIDX_TOMBSTONES` (JSON file `[["regex","text"],...]`, replaces hints and renames), `MEMIDX_PRIVATE_PREFIXES`,
`MEMIDX_OWN_PREFIX`, `MEMIDX_PRIVATE_MODE`, `MEMIDX_MEMORY_PREFIX` (alias `MEMIDX_GEDAECHTNIS_PREFIX`),
`MEMIDX_LOG`, `MEMIDX_VECTORS` (`0`/`1`), `MEMIDX_MODEL_CACHE`, `MEMIDX_EMBEDDER`, `MEMIDX_THREADS`
(onnxruntime threads), `MEMIDX_GRAPH_WORKERS`, `MEMIDX_RRF_BM`, `MEMIDX_RRF_VEC`, `MEMIDX_RRF_DOC`,
`MEMIDX_RRF_DOCTOP`, `MEMIDX_LINKS` (`0` = off), `MEMIDX_LINKS_MAX`, `MEMIDX_RECENCY` (`1` = on),
`MEMIDX_MCP_HOST`, `MEMIDX_MCP_PORT`, `MEMIDX_MCP_CLIENT`, `MEMIDX_MCP_TRANSPORT`, `MEMIDX_INTERVAL`,
`MEMIDX_CLIENT` and `MEMIDX_URL` (CLI `search`).

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
memidx has no authentication; keep it on `127.0.0.1` (the default) or put an authenticating proxy in front.

### Custom embedder

Any object with `embed_documents(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]` works.
Set it in Python with `memidx.embed.set_embedder(obj)` or by config (`embedder = "pkg.module:factory"`, called with
the config or without arguments). An optional `name` attribute is stored in the index; when it changes, `update`
discards the old vectors. The tests use this hook with a small deterministic fake embedder.

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
memidx [--config F] [--corpus DIR] [--db F] COMMAND
  init [--force]                                  write a commented memidx.toml into the corpus
  update [--no-embed]                             index (incremental); embeds when vectors are available
  search QUERY... [-k N] [--budget T] [--zone Z] [--list | --json] [--all] [--client ID] [--url URL]
  get PATH[:FROM-TO] [--client ID]
  status
  serve [--host H] [--port P] [--interval S] [--no-auto-update] [--no-embed] [--mcp]
  mcp [--transport stdio|streamable-http] [--host H] [--port P]
  model download [--cache-dir DIR]
```

`search --url http://127.0.0.1:8800` queries a running server instead of the local file. `serve` indexes once at
start, then every `interval` seconds (0 = only at start); `--mcp` also starts the MCP server as a child process.

### HTTP (default `127.0.0.1:8800`)

| Endpoint | Parameters | Result |
|---|---|---|
| `GET /search` | `q`, `k`=3, `budget`=800, `zone`, `mode`=`json`/`text`/`list`, `all`=1, `client` (default `http`) | `{"query", "hints", "hits": [{"path", "section", "lines", "zone", "score", "text", "stale"?, "superseded_by"?, "memory"?}], "tokens"}` |
| `GET /doc` | `path`, `from`=1, `to`=0 (end), `client` | text, 404 for unknown or private paths |
| `GET /related` | `path`, `all`, `client` | list of hits (wikilink neighbours carry `"via": "link"`) |
| `GET /list` | `zone`, `prefix`, `limit`=200 (max 1000), `client` | `{"zones": {...}, "files": [{"path", "zone", "title", "descr", "sections"}]}` |
| `GET /graph` | `limit`=300 (max 1000), `zone`, `score`=1, `client` | `{"nodes": [{"path", "zone", "title", "mtime", "hash"}], "edges": [[a, b(, score)]], "n", "token", "computed_s"}` |
| `GET /status` | | `{"files", "sections", "vectors", "zones", "normalization", "embedder"}` |
| `GET /` | `q` | HTML read UI |

Bad numbers give 400. Each request uses its own database connection.

### MCP

`memidx mcp` (stdio, default) or `memidx mcp --transport streamable-http` (`127.0.0.1:8801`). Tools:
`memory_search(query, k=3, budget_tokens=800, zone=None)`, `memory_get(path, from_line=1, to_line=0)`,
`memory_related(path, k=5)`, `memory_status()`. Works with `mcp` 1.x and 2.x.

## Limits

- No authentication or authorization. The client id is self-declared.
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

## License

MIT, see `LICENSE`. Third-party notices: `THIRD_PARTY_NOTICES.md`. The embedding model is not part of this
repository and is distributed by Google under the Gemma Terms of Use.
