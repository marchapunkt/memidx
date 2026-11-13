# Changelog

## 0.2.0

Security hardening after a review. memidx is now meant to be safe to run on a checkout you do not fully trust;
see `SECURITY.md` and the "Security model" section of the README.

### Breaking changes

- **The repository's `memidx.toml` is untrusted.** It may only set `languages`, `timezone`, `exclude`,
  `default_zone`, `zones`, `vector_min_weight`, `hints`, `renames`, `memory_entries_prefix` and `ranking`. All
  other keys there (`embedder`, `db`, `log_path`, `model_cache`, `vectors`, `server`, `private_prefixes`,
  `own_client_prefix`, `private_mode`, `graph_workers`, unknown keys) are ignored with one notice. Move them to the
  new user config (`memidx init --user`), a `--config` file or `MEMIDX_*` variables. Unknown keys in that file are
  no longer an error.
- **New default index location** outside the repository: `<user cache>/memidx/index/<name>-<hash>/index.db`. An
  existing `<repo>/.memidx/index.db` is no longer read; run `memidx update` once to build the new index, then
  delete the old directory. Explicit `db`, `MEMIDX_DB` and `--db` still work.
- **Vectors are opt-in and need an explicit model.** `vectors` now defaults to `false`. With `vectors = true` you
  must set `model = "arctic-embed-m-v2.0"` (or an `embedder` hook) in a trusted source; otherwise `update`,
  `search`, `serve` and `mcp` exit with code 2.
- **New embedding model, EmbeddingGemma removed.** The only preset is now `arctic-embed-m-v2.0`
  (Snowflake/snowflake-arctic-embed-m-v2.0, int8 ONNX, 768 dimensions, Apache License 2.0). EmbeddingGemma is
  gone because the Gemma Terms of Use are not an OSI-approved license; `model = "embeddinggemma-300m"` is a
  configuration error that names the replacement. Vectors of a 0.1.0 index are discarded and embedded again on
  the next `memidx update` (download the new model first). On a private corpus of 3,453 Markdown files (40
  natural-language questions, 25 keyword queries) the new model found the expected file as often as
  EmbeddingGemma, within one question, and embedded about twice as fast.
- **`memidx model download` needs the preset name.** A preset whose license is not OSI-approved would also need
  `--accept-license`; no shipped preset has such a license.
- The 0.1.0 module constants in `memidx.embed` (`REPOSITORY`, `REVISION`, `MODEL_FILE`, `MANIFEST`, `DIMENSION`,
  `MAX_TOKENS`, `QUERY_PROMPT`, `DOCUMENT_PROMPT`, `LICENSE_NOTICE`) and `GemmaEmbedder` are removed; use
  `memidx.embed.PRESETS` and `OnnxEmbedder`.
- **`MEMIDX_URL` is no longer read.** Remote search needs an explicit `--url`.
- **Exit codes changed.** 1 = no hit (was 0), 2 = usage or configuration error including unknown zone and a query
  that is empty after normalization (was 0 with no hits), 3 = no usable index (was 2), git failure in a work tree,
  or server unreachable.
- **A failing `git ls-files` inside a git work tree is an error** (exit 3) instead of a silent walk over all files.
- **git's ownership check stays on.** memidx no longer passes `-c safe.directory=*`. Use `git_safe_directory = true`,
  `MEMIDX_GIT_SAFE_DIRECTORY=1` or `--trust-repo-owner` for a repository owned by another user (the Docker image
  sets the variable).
- **`search`, `status`, `get`, `serve` (without auto-update), `mcp` and HTTP requests never create a database**;
  they open it read-only. `/search` returns 400 for `empty_query`/`unknown_zone` and every endpoint returns 503
  without a usable index.
- Search results that are empty carry `reason` and `message`; results may carry `private_excluded: true`. Text
  output of an empty result is now `(no hits) [reason] message`.
- `memidx status` prints the config files as a list with their trust level (`config` was a single string) and an
  `environment` block; it exits with 3 when there is no usable index.
- Python API: `embed.model_dir`, `embed.missing_files` and `embed.download_model` take a preset argument;
  `core.open_db` no longer creates a missing file.

### Added

- Per-user config file (`$XDG_CONFIG_HOME/memidx/config.toml`, `~/.config/memidx/config.toml`, Windows
  `%APPDATA%\memidx\config.toml`) and the layering defaults < user config < repository file < `--config` /
  `MEMIDX_CONFIG` < environment < flags.
- Global flags `--no-corpus-config` (also `MEMIDX_NO_CORPUS_CONFIG=1`), `--isolated`/`-I` (ignore all `MEMIDX_*`
  variables and the user config) and `--trust-repo-owner`.
- `memidx init --user`; `memidx init` inside the repository writes only the repository-safe keys.
- `memidx model list`, the model preset table (`embed.PRESETS`) and the trusted settings `model`
  (`MEMIDX_MODEL`) and `git_safe_directory`.
- One notice when `MEMIDX_CONFIG`, `MEMIDX_DB`, `MEMIDX_LOG`, `MEMIDX_MODEL_CACHE` or `MEMIDX_EMBEDDER` changes
  where data is read or written (names only); `status` lists set and effective `MEMIDX_*` names.
- `update` prints whether it indexed git-tracked files or walked the tree.
- POSIX: index directories 0700, database, `-wal`, `-shm` and query log 0600.
- README sections "A pointer, not proof" and "Security model"; `SECURITY.md`.

### Fixed

- git calls pass `-c core.fsmonitor=false -c core.untrackedCache=false` and `GIT_OPTIONAL_LOCKS=0`, so a
  repository-local git config cannot start programs through them.
- Hint regexes from the repository file are length- and count-limited; an invalid one is skipped with a notice.
- The example config placed top-level keys after a `[table]` header, so `vector_min_weight` ended up inside
  `[default_zone]` and was silently ignored.

## 0.1.0

Initial release: section-level hybrid search (SQLite FTS5 plus optional ONNX embeddings) over the Markdown files of
a git repository, with zones, hints, wikilinks, memory entries, private prefixes, CLI, HTTP server and MCP server.
