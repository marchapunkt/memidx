# Security policy

## Threat model

memidx is meant to be safe to run on a checkout you do not fully trust, for example a cloned repository or a
branch someone else wrote. Running `memidx update` or `memidx search` in such a checkout must not run code from
it, write outside the places you chose, change who may see private paths, or redirect your queries.

**Trusted** (may set every option): command-line flags, `MEMIDX_*` environment variables, the per-user config file
(`$XDG_CONFIG_HOME/memidx/config.toml`, `~/.config/memidx/config.toml`, Windows `%APPDATA%\memidx\config.toml`)
and a file passed with `--config` or `MEMIDX_CONFIG`.

**Untrusted:** everything inside the indexed repository. This includes the Markdown files and
`<repo>/memidx.toml`, which may only set search-related keys (`languages`, `timezone`, `exclude`,
`default_zone`, `zones`, `vector_min_weight`, `hints`, `renames`, `memory_entries_prefix`, `ranking`). Keys that
import code (`embedder`), choose write locations (`db`, `log_path`, `model_cache`), open ports (`server`), load a
model (`vectors`, `model`), relax git (`git_safe_directory`) or change private-path visibility (`private_*`) are
ignored there with a notice. `--no-corpus-config` ignores the file completely; `--isolated` ignores the
environment and the user config.

Other measures: the index lives outside the repository in your user cache directory (POSIX: directories 0700,
files 0600; Windows: the ACL of your profile directory); read commands never create a database; git is called with
fsmonitor and the untracked cache disabled and with its ownership check on; empty results say why they are empty.

## Out of scope

- **No authentication** on the HTTP and MCP servers. Anyone who can reach the port can read the whole index and
  can send any self-declared client id, including one that sees private paths. Keep the servers on `127.0.0.1`
  (the default) or put an authenticating proxy in front.
- Private prefixes protect against accidental disclosure to agents that use the default client id. They are not
  access control.
- Secret scrubbing is a heuristic. Do not rely on it to keep credentials out of the index.
- Regular expressions in `hints` from the repository file are limited in length and number, but a crafted pattern
  can still be slow on a query (regular-expression denial of service). Use `--no-corpus-config` for repositories
  you distrust strongly.
- Model files: memidx verifies size and SHA-256 of the files of a preset after download. The model's license is
  your responsibility; see `memidx model list`.
- A trusted source is trusted completely: an `embedder` hook in your user config runs that code.
- **`python -m memidx` inside an untrusted checkout.** With `-m`, Python itself puts the current directory first
  on its module search path, so a file such as `json.py` in the repository is imported before memidx runs. memidx
  cannot prevent this. Use the installed `memidx` command, or `python -P -m memidx` (or `PYTHONSAFEPATH=1`).

## Reporting a vulnerability

Please report security problems through a private security advisory on the GitHub repository ("Security" tab,
"Report a vulnerability"). For problems that do not need confidentiality, a normal GitHub issue is fine. Include
the memidx version, your platform, and steps to reproduce.
