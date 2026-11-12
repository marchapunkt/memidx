"""memidx engine: section index over all Markdown files of a corpus.

Principles
- Files (git) are the truth; the index is derived and can be rebuilt at any time.
- Hits are SECTIONS (source, heading path, line range), limited by a token budget.
- Hybrid: SQLite FTS5 (normalized text) + optional vectors (only zones with weight >= vector_min_weight),
  fused by reciprocal rank fusion; zone weights scale the scores.
- Hints: configured notes (e.g. retired or renamed systems) are put in front of the answer.
- Incremental: one content hash per file; only changed files are re-indexed, vectors of unchanged
  sections are reused.
"""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache

from . import embed as _embed
from .config import load_config
from .lang import Normalizer

# ------------------------------------------------------------------ configuration
_CFG = None
_GEN = 0          # incremented by configure(); keys caches that depend on the configuration
_NORMS = {}
_HINTS = {}


def configure(cfg):
    """Set the active configuration (one per process). Clears caches that depend on it."""
    global _CFG, _GEN
    _CFG = cfg
    _GEN += 1
    _HINTS.clear()
    _GRAPH_CACHE.clear()
    invalidate_cache()
    return cfg


def get_config():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def normalizer():
    langs = get_config().languages
    n = _NORMS.get(langs)
    if n is None:
        n = _NORMS[langs] = Normalizer(langs)
    return n


def terms(text, drop_stop=True):
    return normalizer().terms(text, drop_stop)


def zone_of(path):
    return get_config().zone_of(path)


# ------------------------------------------------------------------ secret scrubbing
SCRUB_VERSION = "4"   # 4: mask text "[redacted]" (the original used a German word)
MASK = "[redacted]"
_SECRET = re.compile(r"(?i)(passwor[dt]|kennwort|token|secret|api[_-]?key|private[_-]?key)(\s*[:=]\s*)(\S{6,})")
_LONG = re.compile(r"(?<![A-Za-z0-9+/_.-])(?=[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/_.-]))(?=[A-Za-z0-9+/=]*\d)(?=[A-Za-z0-9+/=]*[a-z])(?=[A-Za-z0-9+/=]*[A-Z])[A-Za-z0-9+/]{40,}={0,2}")


def scrub(text):
    """Masks credential values (key=value) and long key-like strings before indexing and embedding."""
    text = _SECRET.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    return _LONG.sub(MASK, text)


# ------------------------------------------------------------------ chunking
FM = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def parse_frontmatter(text):
    m = FM.match(text)
    fm = {}
    if not m:
        return fm, 0
    for line in m.group(1).split("\n"):
        if ":" in line and not line.startswith((" ", "\t", "-")):
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"')
    return fm, m.group(0).count("\n")


# Frontmatter keys: the German keys of the original first, English aliases second.
FM_KEYS = {
    "ersetzt_durch": ("ersetzt_durch", "superseded_by"),
    "gueltig_bis": ("gueltig_bis", "valid_until"),
    "art": ("art", "kind"),
    "gueltig_ab": ("gueltig_ab", "valid_from"),
    "bestaetigt_am": ("bestaetigt_am", "confirmed_on"),
}


def fm_value(fm, column):
    for k in FM_KEYS[column]:
        if fm.get(k):
            return fm[k]
    return ""


def sections(text, max_chars=3200):
    """(frontmatter, [(heading_path, line_start, line_end, body)]) with 1-based lines."""
    lines = text.split("\n")
    fm, skip = parse_frontmatter(text)
    stack, cur, start, in_fence, out = [], [], skip + 1, False, []

    def flush(end):
        body = "\n".join(cur).strip()
        if not body:
            return
        hp = " > ".join(h for _, h in stack) or "(start)"
        if len(body) > max_chars:
            parts, buf = [], ""
            for para in re.split(r"\n\s*\n", body):
                if len(buf) + len(para) > max_chars and buf:
                    parts.append(buf); buf = ""
                buf += ("\n\n" if buf else "") + para
            if buf:
                parts.append(buf)
            for i, part in enumerate(parts):
                out.append((hp + (f" (part {i+1})" if len(parts) > 1 else ""), start, end, part))
        else:
            out.append((hp, start, end, body))

    for i, line in enumerate(lines[skip:], start=skip + 1):
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            flush(i - 1)
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            cur, start = [], i + 1
        else:
            cur.append(line)
    flush(len(lines))
    return fm, out


# ------------------------------------------------------------------ vectors (optional)
_VM = {}
_LOCK = threading.Lock()   # serializes index updates inside one process


def _vec_min():
    return get_config().vector_min_weight


def _norm(v):
    import numpy as np
    a = np.asarray(v, dtype="float32"); n = np.linalg.norm(a)
    return a / n if n else a


def invalidate_cache():
    _VM.clear()


def get_embedder():
    return _embed.get_embedder(get_config())


def vec_matrix(db):
    import numpy as np
    key = (_db_file(db), db.execute("SELECT count(*) FROM vec").fetchone()[0])
    if _VM.get("key") != key:
        rows = db.execute("SELECT id, v FROM vec").fetchall()
        if not rows:
            _VM.update(key=key, ids=None, mat=None)
        else:
            _VM.update(key=key, ids=np.array([r[0] for r in rows]),
                       mat=np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows]))
    return _VM["ids"], _VM["mat"]


def _check_embedder(db, emb, log):
    """Vectors of a different embedder are not comparable: drop them when the embedder changes."""
    name = _embed.embedder_name(emb)
    old = db.execute("SELECT v FROM meta WHERE k='embedder'").fetchone()
    if old and old[0] != name:
        db.execute("DELETE FROM vec"); db.execute("DELETE FROM dvec")
        log(f"Embedder changed ({old[0]} -> {name}): stored vectors discarded")
        invalidate_cache()
    db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('embedder', ?)", (name,))
    db.commit()


def embed_missing(db, log=print, batch=16):
    rows = db.execute("SELECT c.id, c.hpath, c.body FROM chunks c JOIN docs d ON d.id=c.doc "
                      "WHERE d.prior >= ? AND c.id NOT IN (SELECT id FROM vec) ORDER BY c.id", (_vec_min(),)).fetchall()
    if not rows:
        return 0
    emb, t0 = get_embedder(), time.time()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vs = emb.embed_documents([(r[1] + " - " + r[2])[:2000] for r in part])
        for r, v in zip(part, vs):
            db.execute("INSERT OR REPLACE INTO vec(id, v) VALUES(?,?)", (r[0], _norm(v).tobytes()))
        if (i // batch) % 10 == 0:
            db.commit(); log(f"  vectors {i+len(part)}/{len(rows)} ({time.time()-t0:.0f}s)")
    db.commit()
    log(f"New vectors: {len(rows)} sections in {time.time()-t0:.0f}s")
    return len(rows)


DOC_VEC_CHARS = 800    # document vector: title, description and the start of the document


def doc_vec_text(title, descr, bodies):
    """Embedded text of a document; bodies = section texts in document order (only the start counts)."""
    head = (title or "") + (" - " + descr if descr else "")
    parts, n = [], 0
    for b in bodies:
        parts.append(b); n += len(b)
        if n >= DOC_VEC_CHARS:
            break
    return (head + " - " + " ".join(parts))[:DOC_VEC_CHARS]


def embed_missing_docs(db, log=print, batch=16):
    """One vector per document (vector zones only): catches questions that paraphrase a topic without any
    single section matching. Recomputed only when the embedded text changed."""
    db.execute("DELETE FROM dvec WHERE doc NOT IN (SELECT id FROM docs WHERE prior >= ?)", (_vec_min(),))
    db.commit()   # leave no open write transaction, even when nothing is embedded
    todo = []
    for did, title, descr in db.execute("SELECT id, title, descr FROM docs WHERE prior >= ?", (_vec_min(),)).fetchall():
        text = doc_vec_text(title, descr, (b for (b,) in db.execute("SELECT body FROM chunks WHERE doc=? ORDER BY id", (did,))))
        h = _sha(text)
        old = db.execute("SELECT h FROM dvec WHERE doc=?", (did,)).fetchone()
        if not old or old[0] != h:
            todo.append((did, h, text))
    if not todo:
        return 0
    emb, t0 = get_embedder(), time.time()
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vs = emb.embed_documents([t[2] for t in part])
        for t, v in zip(part, vs):
            db.execute("INSERT OR REPLACE INTO dvec(doc, h, v) VALUES(?,?,?)", (t[0], t[1], _norm(v).tobytes()))
        if (i // batch) % 10 == 0:
            db.commit(); log(f"  document vectors {i+len(part)}/{len(todo)} ({time.time()-t0:.0f}s)")
    db.commit()
    log(f"New document vectors: {len(todo)} in {time.time()-t0:.0f}s")
    return len(todo)


def doc_vec_matrix(db):
    try:
        # key from the text hashes: also detects replaced vectors at an unchanged count
        key = (_db_file(db),) + tuple(db.execute("SELECT count(*), coalesce(group_concat(h), '') FROM (SELECT h FROM dvec ORDER BY doc)").fetchone())
    except sqlite3.OperationalError:   # index without document vectors: ranking as before
        return None, None
    import numpy as np
    if _VM.get("dkey") != key:
        rows = db.execute("SELECT doc, v FROM dvec").fetchall()
        if not rows:
            _VM.update(dkey=key, dids=None, dmat=None)
        else:
            _VM.update(dkey=key, dids=np.array([r[0] for r in rows]),
                       dmat=np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows]))
    return _VM["dids"], _VM["dmat"]


# ------------------------------------------------------------------ index (incremental)
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, path TEXT UNIQUE, zone TEXT, prior REAL, title TEXT, status TEXT, descr TEXT, hash TEXT, mtime REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc INTEGER, hpath TEXT, l1 INTEGER, l2 INTEGER, body TEXT, chash TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(h, b, content='', tokenize='unicode61');
CREATE TABLE IF NOT EXISTS vec(id INTEGER PRIMARY KEY, v BLOB);
CREATE TABLE IF NOT EXISTS dvec(doc INTEGER PRIMARY KEY, h TEXT, v BLOB);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS linkroh(doc INTEGER, ziel TEXT);
CREATE INDEX IF NOT EXISTS linkroh_doc ON linkroh(doc);
CREATE TABLE IF NOT EXISTS links(src TEXT, dst TEXT, UNIQUE(src, dst));
CREATE INDEX IF NOT EXISTS links_dst ON links(dst);
"""

# Wikilinks: [[target]], [[target|label]], [[target#section]]; code blocks do not count.
LINKS_VERSION = "1"
_WIKILINK = re.compile(r"\[\[([^\[\]|#\n]+)(?:#[^\[\]|\n]*)?(?:\|[^\[\]\n]*)?\]\]")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def wikilinks(text):
    """Raw target names of the wikilinks of a file (no duplicates, order kept)."""
    text = _CODE.sub("", text)
    out = []
    for m in _WIKILINK.finditer(text):
        z = m.group(1).strip()
        if z and z not in out:
            out.append(z)
    return out


def _lname(s):
    s = s.strip().replace("\\", "/").lower()
    return s[:-3] if s.endswith(".md") else s


def resolve_links(db):
    """Resolves raw wikilink targets against path (suffix), file name and title/frontmatter name; ambiguous and
    unknown targets and self links are ignored. Rebuilds the links table completely (cheap)."""
    docs = db.execute("SELECT id, path, title FROM docs").fetchall()
    by_path, by_base, by_title = {}, {}, {}
    for did, p, t in docs:
        lp = _lname(p)
        by_path[lp] = p
        by_base.setdefault(lp.rsplit("/", 1)[-1], set()).add(p)
        if t:
            by_title.setdefault(_lname(t), set()).add(p)
    paths = {did: p for did, p, _ in docs}
    db.execute("DELETE FROM links")
    for did, ziel in db.execute("SELECT doc, ziel FROM linkroh").fetchall():
        src = paths.get(did)
        z = _lname(ziel)
        if src is None or not z:
            continue
        cands = set()
        if z in by_path:
            cands = {by_path[z]}
        elif "/" in z:
            cands = {p for lp, p in by_path.items() if lp.endswith("/" + z)}
        else:
            cands = by_base.get(z, set()) or by_title.get(z, set())
        if len(cands) == 1:
            dst = next(iter(cands))
            if dst != src:
                db.execute("INSERT OR IGNORE INTO links(src, dst) VALUES(?,?)", (src, dst))


def _fts_texts(rel, descr, hpath, body):
    return " ".join(terms(hpath + " " + os.path.basename(rel) + " " + descr, False)), " ".join(terms(body, False))


def _sha(s):
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _drop_doc_chunks(db, doc_id, rel, descr):
    """Removes the sections of a file (FTS/vectors); returns reusable vectors {chash: blob}."""
    reuse = {}
    for cid, hp, body, ch in db.execute("SELECT id, hpath, body, chash FROM chunks WHERE doc=?", (doc_id,)).fetchall():
        h, b = _fts_texts(rel, descr, hp, body)
        db.execute("INSERT INTO fts(fts, rowid, h, b) VALUES('delete', ?, ?, ?)", (cid, h, b))
        v = db.execute("SELECT v FROM vec WHERE id=?", (cid,)).fetchone()
        if v and ch:
            reuse[ch] = v[0]
        db.execute("DELETE FROM vec WHERE id=?", (cid,))
    db.execute("DELETE FROM chunks WHERE doc=?", (doc_id,))
    return reuse


def _rebuild_fts(db):
    """Recreates the contentless FTS table from the stored sections (normalization rules changed)."""
    db.execute("DROP TABLE IF EXISTS fts")
    db.execute("CREATE VIRTUAL TABLE fts USING fts5(h, b, content='', tokenize='unicode61')")
    n = 0
    for cid, path, descr, hp, body in db.execute(
            "SELECT c.id, d.path, d.descr, c.hpath, c.body FROM chunks c JOIN docs d ON d.id=c.doc").fetchall():
        hh, bb = _fts_texts(path, descr or "", hp, body)
        db.execute("INSERT INTO fts(rowid,h,b) VALUES(?,?,?)", (cid, hh, bb))
        n += 1
    return n


def _excluded(rel, exclude):
    return any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in exclude)


def iter_md(corpus, exclude=None):
    """Relative paths of all Markdown files: in a git checkout only tracked files (git ls-files), else the tree."""
    exclude = get_config().exclude if exclude is None else exclude
    if os.path.exists(os.path.join(corpus, ".git")):     # directory, or file for worktrees/submodules
        try:
            out = subprocess.run(["git", "-c", "safe.directory=*", "-C", corpus, "ls-files", "-z", "--", "*.md"],
                                 capture_output=True, check=True).stdout
            for rel in sorted(out.decode("utf-8", "ignore").split("\0")):
                if rel and not _excluded(rel, exclude):
                    yield rel
            return
        except (OSError, subprocess.CalledProcessError):
            pass
    for dp, dn, fn in os.walk(corpus):
        dn[:] = sorted(d for d in dn if d not in (".git", "node_modules", ".memidx") and not d.startswith(".next"))
        rel_dir = os.path.relpath(dp, corpus).replace(os.sep, "/")
        if rel_dir != "." and _excluded(rel_dir, exclude):
            dn[:] = []
            continue
        for f in sorted(fn):
            if f.endswith(".md"):
                yield os.path.relpath(os.path.join(dp, f), corpus).replace(os.sep, "/")


def _connect_rw(dbpath):
    d = os.path.dirname(os.path.abspath(dbpath))
    os.makedirs(d, exist_ok=True)
    db = sqlite3.connect(dbpath)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    for col in ("ersetzt_durch", "gueltig_bis", "art", "gueltig_ab", "bestaetigt_am"):   # validity + memory entries
        try:
            db.execute(f"ALTER TABLE docs ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    return db


def update(corpus, dbpath, embed=False, log=print):
    cfg = get_config()
    db = _connect_rw(dbpath)
    try:
        return _update(db, corpus, dbpath, embed, log, cfg)
    finally:
        db.close()
        invalidate_cache()


def _update(db, corpus, dbpath, embed, log, cfg):
    ver = db.execute("SELECT v FROM meta WHERE k='scrub'").fetchone()
    if not ver or ver[0] != SCRUB_VERSION:   # rules changed: re-chunk all files (vectors of equal sections stay)
        db.execute("UPDATE docs SET hash=''")
        db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('scrub', ?)", (SCRUB_VERSION,))
        db.commit()
    sig = normalizer().signature
    nver = db.execute("SELECT v FROM meta WHERE k='norm'").fetchone()
    if not nver or nver[0] != sig:            # languages or normalization rules changed: rebuild full text
        n = _rebuild_fts(db)
        db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('norm', ?)", (sig,))
        db.commit()
        if n:
            log(f"Normalization changed ({nver[0] if nver else 'none'} -> {sig}): full-text index rebuilt ({n} sections)")
    lver = db.execute("SELECT v FROM meta WHERE k='links'").fetchone()
    links_backfill = not lver or lver[0] != LINKS_VERSION   # older indexes: read wikilinks of all files once
    links_changed = links_backfill
    existing = {p: (i, h, d) for i, p, h, d in db.execute("SELECT id, path, hash, descr FROM docs")}
    zones_db = {p: (z, pr) for p, z, pr in db.execute("SELECT path, zone, prior FROM docs")}
    rezoned = 0
    seen, changed, same, n_chunks, t0 = set(), 0, 0, 0, time.time()
    for rel in iter_md(corpus, cfg.exclude):
        full = os.path.join(corpus, rel)
        try:
            with open(full, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        seen.add(rel)
        h = _sha(text)
        if rel in existing and existing[rel][1] == h:
            zn = zone_of(rel)
            if zones_db.get(rel) != zn:      # zone table changed: update the zone without re-indexing
                db.execute("UPDATE docs SET zone=?, prior=? WHERE id=?", (zn[0], zn[1], existing[rel][0]))
                rezoned += 1
            if links_backfill:
                db.execute("DELETE FROM linkroh WHERE doc=?", (existing[rel][0],))
                db.executemany("INSERT INTO linkroh(doc, ziel) VALUES(?,?)", [(existing[rel][0], z) for z in wikilinks(text)])
            same += 1
            continue
        text = scrub(text)
        fm, secs = sections(text)
        zone, prior = zone_of(rel)
        title = fm.get("name") or fm.get("title") or (secs[0][0] if secs else rel)
        descr = fm.get("description", "")
        vals = (zone, prior, title, fm.get("status", ""), descr, h, os.path.getmtime(full),
                fm_value(fm, "ersetzt_durch"), fm_value(fm, "gueltig_bis"), fm_value(fm, "art"),
                fm_value(fm, "gueltig_ab"), fm_value(fm, "bestaetigt_am"))
        reuse = {}
        if rel in existing:
            did = existing[rel][0]
            reuse = _drop_doc_chunks(db, did, rel, existing[rel][2] or "")
            db.execute("UPDATE docs SET zone=?, prior=?, title=?, status=?, descr=?, hash=?, mtime=?, ersetzt_durch=?, gueltig_bis=?, "
                       "art=?, gueltig_ab=?, bestaetigt_am=? WHERE id=?", vals + (did,))
        else:
            did = db.execute("INSERT INTO docs(zone,prior,title,status,descr,hash,mtime,ersetzt_durch,gueltig_bis,art,gueltig_ab,bestaetigt_am,path) "
                             "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", vals + (rel,)).lastrowid
        db.execute("DELETE FROM linkroh WHERE doc=?", (did,))
        old_dv = db.execute("SELECT h FROM dvec WHERE doc=?", (did,)).fetchone()
        if old_dv and old_dv[0] != _sha(doc_vec_text(title, descr, [b for _, _, _, b in secs])):
            db.execute("DELETE FROM dvec WHERE doc=?", (did,))   # embedded text changed: drop the vector (never stale)
        db.executemany("INSERT INTO linkroh(doc, ziel) VALUES(?,?)", [(did, z) for z in wikilinks(text)])
        links_changed = True
        for hp, l1, l2, body in secs:
            ch = _sha(hp + "\n" + body)
            cid = db.execute("INSERT INTO chunks(doc,hpath,l1,l2,body,chash) VALUES(?,?,?,?,?,?)", (did, hp, l1, l2, body, ch)).lastrowid
            hh, bb = _fts_texts(rel, descr, hp, body)
            db.execute("INSERT INTO fts(rowid,h,b) VALUES(?,?,?)", (cid, hh, bb))
            if ch in reuse:
                db.execute("INSERT OR REPLACE INTO vec(id, v) VALUES(?,?)", (cid, reuse[ch]))
            n_chunks += 1
        changed += 1
    gone = [p for p in existing if p not in seen]
    for p in gone:
        did, _, descr = existing[p]
        _drop_doc_chunks(db, did, p, descr or "")
        db.execute("DELETE FROM docs WHERE id=?", (did,))
        db.execute("DELETE FROM linkroh WHERE doc=?", (did,))
        db.execute("DELETE FROM dvec WHERE doc=?", (did,))
        links_changed = True
    if links_changed or changed:
        resolve_links(db)
    if links_backfill:
        db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('links', ?)", (LINKS_VERSION,))
    db.commit()
    if rezoned:
        log(f"Zones reassigned: {rezoned} files")
    log(f"Index: {changed} changed/new ({n_chunks} sections), {same} unchanged, {len(gone)} removed, {time.time()-t0:.1f}s")
    new_vec = 0
    if embed and cfg.vectors:
        try:
            emb = get_embedder()
            _check_embedder(db, emb, log)
            new_vec = embed_missing(db, log)
            embed_missing_docs(db, log)
        except _embed.VectorsUnavailable as e:
            _embed.notice(str(e))
    elif embed:
        _embed.notice("vectors disabled by configuration; using full-text search only")
    if changed or gone:
        db.execute("INSERT INTO fts(fts) VALUES('optimize')")
        db.commit()
    total = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    nv = db.execute("SELECT count(*) FROM vec").fetchone()[0]
    log(f"Total: {db.execute('SELECT count(*) FROM docs').fetchone()[0]} files, {total} sections, {nv} vectors, "
        f"DB {os.path.getsize(dbpath)/1e6:.1f} MB")
    return {"changed": changed, "unchanged": same, "removed": len(gone), "new_vectors": new_vec}


# ------------------------------------------------------------------ hints
def _compiled_hints():
    cfg = get_config()
    if _GEN not in _HINTS:
        _HINTS.clear()
        _HINTS[_GEN] = [(re.compile(p), t) for p, t in cfg.all_hints()]
    return _HINTS[_GEN]


def hints(q):
    ql = fold_query(q)
    return [h for pat, h in _compiled_hints() if pat.search(ql)]


def fold_query(q):
    from .lang import fold
    return fold(q)


# ------------------------------------------------------------------ private paths
# Paths under private_prefixes are visible only to clients whose self-declared client id starts with
# own_client_prefix. This is a convention against accidental disclosure, not access control.
def private_prefixes():
    return tuple(get_config().private_prefixes)


def is_private(path):
    pre = private_prefixes()
    return bool(pre) and path.startswith(pre)


def sees_private(client):
    cfg = get_config()
    if cfg.private_mode == "all":
        return True
    return bool(client) and bool(cfg.own_client_prefix) and str(client).startswith(cfg.own_client_prefix)


def private_sql(sees, alias="d"):
    """(SQL clause, parameters) that excludes private paths; empty when the client may see them."""
    pre = private_prefixes()
    if sees or not pre:
        return "", ()
    esc = lambda s: s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    clause = " AND NOT (" + " OR ".join(f"{alias}.path LIKE ? ESCAPE '\\'" for _ in pre) + ")"
    return clause, tuple(esc(p) + "%" for p in pre)


# ------------------------------------------------------------------ query log
_LOG_LOCK = threading.Lock()


def log_query(client, query, k, budget, zone, res, secs):
    """Appends one JSON line per query (time, client, query, hit paths, tokens). Errors are ignored."""
    path = get_config().log_path
    if not path:
        return
    try:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "client": client or "?", "q": query, "k": k, "budget": budget, "zone": zone,
               "n": len(res.get("hits", [])), "tok": res.get("tokens", 0), "hint": len(res.get("hints", [])),
               "paths": [t["path"] + ":" + t["lines"] for t in res.get("hits", [])], "secs": round(secs, 3),
               "private": sees_private(client)}
        with _LOG_LOCK:
            if os.path.exists(path) and os.path.getsize(path) > 5_000_000:
                os.replace(path, path + ".1")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------ search
def open_db(dbpath):
    db = sqlite3.connect(dbpath, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


@lru_cache(maxsize=100_000)   # the same section lines are re-scored by many queries
def _line_terms_for(sig, ln):
    return frozenset(terms(ln, True))


def _line_terms(ln):
    return _line_terms_for(normalizer().signature, ln)


def snippet(body, qterms, max_chars):
    if len(body) <= max_chars:
        return body.strip()
    lines = body.split("\n")
    scored = [(len(_line_terms(ln) & qterms), i) for i, ln in enumerate(lines)]
    best = max(scored)[1] if scored else 0
    lo, hi, size = best, best, len(lines[best])
    while (lo > 0 or hi < len(lines) - 1) and size < max_chars:
        if lo > 0 and (hi >= len(lines) - 1 or size % 2 == 0):
            lo -= 1; size += len(lines[lo]) + 1
        elif hi < len(lines) - 1:
            hi += 1; size += len(lines[hi]) + 1
    out = "\n".join(lines[lo:hi + 1]).strip()
    return ("…\n" if lo > 0 else "") + out[:max_chars] + ("\n…" if hi < len(lines) - 1 else "")


_CHUNK_SQL = ("SELECT c.id, c.hpath, c.l1, c.l2, c.body, d.path, d.zone, d.prior, d.status, d.ersetzt_durch, d.gueltig_bis, d.mtime, "
              "d.art, d.gueltig_ab, d.bestaetigt_am "
              "FROM chunks c JOIN docs d ON d.id=c.doc WHERE c.id IN ({})")


def search(db, query, k=3, budget_tokens=800, zone=None, per_file=2, sec_chars=1400, hybrid=True, client=None,
           private_ok=None, include_invalid=False, today=None):
    t_start = time.time()
    ok = sees_private(client) if private_ok is None else private_ok
    res = _search(db, query, k, budget_tokens, zone, per_file, sec_chars, hybrid, ok, include_invalid, today)
    if client:
        log_query(client, query, k, budget_tokens, zone, res, time.time() - t_start)
    return res


def stale(r, today=None):
    """A document is stale with status superseded, with superseded_by set, or with valid_until (YYYY-MM-DD) in
    the past. Stale hits are down-weighted and marked, not hidden."""
    if r["status"] == "superseded" or r["ersetzt_durch"]:
        return True
    gb = (r["gueltig_bis"] or "").strip()
    return bool(gb) and gb < (today or get_config().today()).isoformat()


# ------------------------------------------------------------------ memory entries
# Files under memory_entries_prefix have their own validity rule. Order of reasons: format, revoked,
# superseded, not_yet_valid, expired, else valid. German and English status values are accepted.
_MEM_STATUS = {"bestaetigt": "confirmed", "confirmed": "confirmed", "widerrufen": "revoked", "revoked": "revoked",
               "ersetzt": "superseded", "superseded": "superseded"}
UNTIL_REVOKED = ("bis_widerruf", "until_revoked")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def memory_prefix():
    return get_config().memory_entries_prefix


def _parse_date(value):
    if not value or not _DATE_RE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validity_text(valid_from, valid_until, confirmed_on):
    """"until revoked" or "until YYYY-MM-DD"; prefixed with "from YYYY-MM-DD " only when valid_from lies
    after confirmed_on (both parseable)."""
    until = "until revoked" if valid_until in UNTIL_REVOKED else f"until {valid_until}"
    f, c = _parse_date(valid_from), _parse_date(confirmed_on)
    if f is not None and c is not None and f > c:
        return f"from {valid_from} {until}"
    return until


def memory_info(r, today):
    """None for hits outside memory_prefix(); else {"kind", "valid", "reason", "valid_from", "valid_until",
    "confirmed_on"}."""
    prefix = memory_prefix()
    if not prefix or not r["path"].startswith(prefix):
        return None
    status = _MEM_STATUS.get(r["status"] or "")
    start = _parse_date(r["gueltig_ab"])
    until_raw = r["gueltig_bis"] or ""
    until = None if until_raw in UNTIL_REVOKED else _parse_date(until_raw)
    if status is None or start is None or (until_raw not in UNTIL_REVOKED and (until is None or until < start)):
        valid, reason = False, "format"
    elif status == "revoked":
        valid, reason = False, "revoked"
    elif status == "superseded" or r["ersetzt_durch"]:
        valid, reason = False, "superseded"
    elif start > today:
        valid, reason = False, "not_yet_valid"
    elif until is not None and until < today:
        valid, reason = False, "expired"
    else:
        valid, reason = True, "valid"
    return {"kind": r["art"] or "", "valid": valid, "reason": reason, "valid_from": r["gueltig_ab"] or "",
            "valid_until": until_raw, "confirmed_on": r["bestaetigt_am"] or ""}


def recency_factor(r):
    """Optional freshness bonus (off by default): up to +15 % for recently changed files, half-life 180 days."""
    if not get_config().ranking.recency or not r["mtime"]:
        return 1.0
    age_days = max(0.0, (time.time() - float(r["mtime"])) / 86400.0)
    return 1.0 + 0.15 * 0.5 ** (age_days / 180.0)


def link_factors(db, cands, top_n=10):
    """Per document a factor 1.0 .. 1+links_max (max 0.10): counts the documents among the best `top_n`
    (by score) that are connected to it by a wikilink (both directions). From 2 such neighbours on there is
    a bonus that grows with their number (4 or more = maximum). Read-only; no edges, empty result."""
    try:
        mx = max(0.0, min(0.10, float(get_config().ranking.links_max)))
    except (TypeError, ValueError):
        mx = 0.10
    top = list(dict.fromkeys(c[1]["path"] for c in cands))[:top_n]
    if mx <= 0 or len(top) < 2:
        return {}
    ph = ",".join("?" * len(top))
    nb = {}
    for a, b in db.execute(f"SELECT src, dst FROM links WHERE src IN ({ph}) OR dst IN ({ph})", (*top, *top)):
        for x, y in ((a, b), (b, a)):
            if y in top and x != y:
                nb.setdefault(x, set()).add(y)
    return {p: 1.0 + mx * min(len(v), 4) / 4.0 for p, v in nb.items() if len(v) >= 2}


RRF_K = 60   # reciprocal rank fusion constant (unchanged)


def _vector_ranks(db, query, rrf, rk):
    """Adds the section-vector and document-vector lists to `rrf`. Silent no-op without vectors."""
    if not _embed.numpy_available():
        return
    ids, mat = vec_matrix(db)
    if mat is None:
        return
    try:
        emb = get_embedder()
    except _embed.VectorsUnavailable as e:
        _embed.notice(str(e))
        return
    import numpy as np
    qv = _norm(emb.embed_query(query))
    sims = mat @ qv
    for rank, j in enumerate(np.argsort(-sims)[:100]):
        cid = int(ids[j]); rrf[cid] = rrf.get(cid, 0.0) + rk.rrf_vector / (RRF_K + rank)
    if rk.rrf_doc > 0:
        dids, dmat = doc_vec_matrix(db)
        if dmat is not None:
            for rank, j in enumerate(np.argsort(-(dmat @ qv))[:rk.rrf_doc_top]):
                for (cid,) in db.execute("SELECT id FROM chunks WHERE doc=?", (int(dids[j]),)):
                    rrf[cid] = rrf.get(cid, 0.0) + rk.rrf_doc / (RRF_K + rank)


def _search(db, query, k, budget_tokens, zone, per_file, sec_chars, hybrid, private_ok=False, include_invalid=False, today=None):
    cfg = get_config()
    rk = cfg.ranking
    today = today or cfg.today()
    qt = terms(query, True)
    uniq = list(dict.fromkeys(qt))
    if not uniq:
        return {"query": query, "hints": hints(query), "hits": [], "tokens": 0}
    match = " OR ".join(f'"{t}"' for t in uniq)
    ids_bm = [r[0] for r in db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 5.0, 1.0) LIMIT 200", (match,))]
    rrf = {}
    for rank, cid in enumerate(ids_bm):
        rrf[cid] = rrf.get(cid, 0.0) + rk.rrf_bm25 / (RRF_K + rank)
    if hybrid and cfg.vectors:
        _vector_ranks(db, query, rrf, rk)
    if not rrf:
        return {"query": query, "hints": hints(query), "hits": [], "tokens": 0}
    rows = {r["id"]: r for r in db.execute(_CHUNK_SQL.format(",".join(map(str, rrf))))}
    cands = []
    for cid, sc in rrf.items():
        r = rows.get(cid)
        if r is None or (zone and r["zone"] != zone) or (not private_ok and is_private(r["path"])):
            continue
        # memory entries: own validity rule instead of stale(); invalid ones are dropped unless include_invalid
        minfo = memory_info(r, today)
        if minfo is not None:
            if not minfo["valid"] and not include_invalid:
                continue
            cands.append((sc * 1000 * r["prior"] * recency_factor(r), r, minfo))
        else:
            old = stale(r, today)
            cands.append((sc * 1000 * r["prior"] * (rk.stale_factor if old else 1.0) * recency_factor(r), r, None))
    cands.sort(key=lambda x: -x[0])
    if rk.links:
        fak = link_factors(db, cands)
        if fak:
            cands = [(sc * fak.get(r["path"], 1.0), r, g) for sc, r, g in cands]
            cands.sort(key=lambda x: -x[0])
    out, per, used, qset = [], {}, 0, set(uniq)
    for score, r, minfo in cands:
        if per.get(r["path"], 0) >= per_file:
            continue
        sn = snippet(r["body"], qset, sec_chars)
        tok = len(sn) // 4
        if out and used + tok > budget_tokens:
            continue
        per[r["path"]] = per.get(r["path"], 0) + 1
        used += tok
        out.append(_hit(r, score, sn, minfo, today))
        if len(out) >= k:
            break
    return {"query": query, "hints": hints(query), "hits": out, "tokens": used}


def _hit(r, score, text, minfo, today, via=None):
    e = {"path": r["path"], "section": r["hpath"], "lines": f'{r["l1"]}-{r["l2"]}', "zone": r["zone"],
         "score": round(score, 2), "text": text}
    if via:
        e["via"] = via
    if minfo is not None:
        e["memory"] = minfo
    elif stale(r, today):
        e["stale"] = True
        if r["ersetzt_durch"]:
            e["superseded_by"] = r["ersetzt_durch"]
    return e


def get_lines(db, path, l1=1, l2=0, max_chars=12000, client=None):
    """Reads a line range of a file from the stored sections (no file access).
    Private paths without own client id: like unknown (None)."""
    if is_private(path) and not sees_private(client):
        return None
    rows = db.execute("SELECT c.l1, c.l2, c.hpath, c.body FROM chunks c JOIN docs d ON d.id=c.doc WHERE d.path=? ORDER BY c.l1", (path,)).fetchall()
    if not rows:
        return None
    parts = []
    for r in rows:
        if l2 and (r["l1"] > l2 or r["l2"] < l1):
            continue
        if r["l2"] < l1:
            continue
        parts.append(f'## {r["hpath"]} (lines {r["l1"]}-{r["l2"]})\n{r["body"]}')
    text = "\n\n".join(parts)
    return text[:max_chars] + ("\n…(truncated)" if len(text) > max_chars else "")


def _link_neighbours(db, path, ok, include_invalid, today, k):
    """Real wikilink neighbours (both directions) as hits in the format of search(); first section of the file."""
    today = today or get_config().today()
    pairs = db.execute("SELECT dst, 0 FROM links WHERE src=? UNION SELECT src, 1 FROM links WHERE dst=?", (path, path)).fetchall()
    sf = get_config().ranking.stale_factor
    out = []
    for nb, _ in pairs:
        if nb == path or (not ok and is_private(nb)):
            continue
        r = db.execute(_CHUNK_SQL.replace("c.id IN ({})", "c.id=(SELECT MIN(c2.id) FROM chunks c2 JOIN docs d2 ON d2.id=c2.doc WHERE d2.path=?)"), (nb,)).fetchone()
        if r is None:
            continue
        g = memory_info(r, today)
        if g is not None and not g["valid"] and not include_invalid:
            continue
        score = 1000 / RRF_K * r["prior"] * (sf if g is None and stale(r, today) else 1.0)
        out.append(_hit(r, score, snippet(r["body"], set(), 400), g, today, via="link"))
    out.sort(key=lambda e: (-e["score"], e["path"]))
    return out[:k]


def related(db, path, k=5, client=None, private_ok=None, include_invalid=False, today=None):
    """Related documents: first the real wikilink edges (both directions), then text similarity (full text)."""
    ok = sees_private(client) if private_ok is None else private_ok
    if is_private(path) and not ok:
        return []
    r = db.execute("SELECT title, descr FROM docs WHERE path=?", (path,)).fetchone()
    if not r:
        return []
    real = _link_neighbours(db, path, ok, include_invalid, today, k)
    if len(real) >= k:
        return real
    res = search(db, f'{r["title"]} {r["descr"]} {os.path.basename(path)}', k=k + 3 + len(real), budget_tokens=400, per_file=1,
                 hybrid=False, private_ok=ok, include_invalid=include_invalid, today=today)
    have = {path} | {e["path"] for e in real}
    return (real + [t for t in res["hits"] if t["path"] not in have])[:k]


# ------------------------------------------------------------------ graph
GRAPH_MAX = 1000
_GRAPH_CACHE = {}
_GRAPH_LOCK = threading.Lock()


def index_token(db):
    """Index state: changes with every content or zone change of a file (also new/removed files)."""
    h = hashlib.sha1()
    for p, z, ha in db.execute("SELECT path, zone, hash FROM docs ORDER BY path"):
        h.update(f"{p}\0{z}\0{ha}\n".encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _db_file(db):
    for _, name, f in db.execute("PRAGMA database_list"):
        if name == "main" and f:
            return f
    return None


def _compute_graph(db, limit, zone, k, with_score, private_ok=False):
    t0 = time.time()
    clause, cparams = private_sql(private_ok)
    rows = db.execute(
        "SELECT d.path, d.zone, d.title, d.mtime, d.hash FROM docs d WHERE (?='' OR d.zone=?)" + clause +
        " ORDER BY d.prior DESC, d.path LIMIT ?", (zone, zone, *cparams, limit)).fetchall()
    nodes = [{"path": r[0], "zone": r[1], "title": r[2], "mtime": int(r[3] or 0), "hash": r[4]} for r in rows]
    paths = [n["path"] for n in nodes]
    dbfile, workers = _db_file(db), max(1, get_config().graph_workers)
    if dbfile and workers > 1 and len(paths) > 1:   # SQLite releases the GIL during queries: one connection per thread
        local = threading.local()
        conns, conns_lock = [], threading.Lock()

        def work(p):
            if not hasattr(local, "db"):
                local.db = open_db(dbfile)
                with conns_lock:
                    conns.append(local.db)
            return related(local.db, p, k, private_ok=private_ok)
        try:
            with ThreadPoolExecutor(workers) as ex:
                rel = list(ex.map(work, paths))
        finally:
            for c in conns:
                c.close()
    else:
        rel = [related(db, p, k, private_ok=private_ok) for p in paths]
    inset, seen, edges = set(paths), {}, []
    for p, hits in zip(paths, rel):
        for t in hits:
            q = t["path"]
            if q == p or q not in inset:
                continue
            key = frozenset((p, q))
            if key in seen:                           # symmetric: each pair once, best score
                seen[key][2] = max(seen[key][2], t["score"])
                continue
            seen[key] = [p, q, t["score"]]
            edges.append(seen[key])
    return {"nodes": nodes, "edges": [e if with_score else e[:2] for e in edges], "n": len(nodes),
            "token": index_token(db), "computed_s": round(time.time() - t0, 2)}


def _graph(db, limit, zone, k, with_score, private_ok=False):
    """Like graph(), but on the given connection."""
    limit, zone = max(1, min(int(limit), GRAPH_MAX)), zone or ""
    key = (_GEN, _db_file(db), limit, zone, k, bool(with_score), bool(private_ok))
    token = index_token(db)
    hit = _GRAPH_CACHE.get(key)
    if hit and hit[0] == token:
        return hit[1]
    with _GRAPH_LOCK:
        hit = _GRAPH_CACHE.get(key)
        if hit and hit[0] == token:
            return hit[1]
        res = _compute_graph(db, limit, zone, k, with_score, private_ok)
        if len(_GRAPH_CACHE) >= 16:
            _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
        _GRAPH_CACHE[key] = (res["token"], res)
        return res


def graph(db, limit=300, zone=None, k=5, with_score=False, client=None):
    """Nodes (like /list) with neighbours (like /related, only inside the node set) as an edge list.
    Cached per index state; computed once at a time, on its own connection."""
    ok = sees_private(client)
    f = _db_file(db)
    if not f:
        return _graph(db, limit, zone, k, with_score, ok)
    own = open_db(f)
    try:
        return _graph(own, limit, zone, k, with_score, ok)
    finally:
        own.close()


def warm_graph(dbpath):
    """Precomputes the default graph (after start and after index changes)."""
    try:
        db = open_db(dbpath)
        try:
            graph(db, 300)
        finally:
            db.close()
    except Exception as e:
        print("[graph] precompute failed:", e, flush=True)


# ------------------------------------------------------------------ text output
def _memory_mark(t):
    """Marker for memory entries without the leading space; None for hits without a `memory` field."""
    g = t.get("memory")
    if g is None:
        return None
    if not g["valid"]:
        return f"[MEMORY invalid: {g['reason']}]"
    kind = "Preference" if g["kind"] in ("praeferenz", "preference") else "Fact"
    text = validity_text(g["valid_from"], g["valid_until"], g["confirmed_on"])
    return f"[MEMORY {kind}, valid {text}, confirmed {g['confirmed_on']}]"


def _stale_mark(t):
    if not t.get("stale"):
        return ""
    return "[STALE" + (f", superseded by {t['superseded_by']}" if t.get("superseded_by") else "") + "]"


def render_text(res):
    parts = [f"[Hint] {h}" for h in res["hints"]]
    for t in res["hits"]:
        mark = _memory_mark(t) or _stale_mark(t)
        mark = "  " + mark if mark else ""
        parts.append(f'== {t["path"]} :: {t["section"]} (lines {t["lines"]}, {t["zone"]}){mark}\n{t["text"]}')
    return "\n\n".join(parts) if parts else "(no hits)"


def render_list(res):
    parts = [f"[Hint] {h}" for h in res["hints"]]
    for i, t in enumerate(res["hits"], 1):
        first = re.sub(r"\s+", " ", t["text"])[:140]
        mark = _memory_mark(t) or _stale_mark(t)
        mark = " " + mark if mark else ""
        parts.append(f"{i}. {t['path']}:{t['lines']} [{t['zone']}]{mark} {t['section']} — {first}")
    return "\n".join(parts) if parts else "(no hits)"


def status(db):
    meta = dict(db.execute("SELECT k, v FROM meta").fetchall())
    return {"files": db.execute("SELECT count(*) FROM docs").fetchone()[0],
            "sections": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "vectors": db.execute("SELECT count(*) FROM vec").fetchone()[0],
            "zones": {r[0]: r[1] for r in db.execute("SELECT zone, count(*) FROM docs GROUP BY zone")},
            "normalization": meta.get("norm", ""),
            "embedder": meta.get("embedder", "")}


def list_files(db, zone="", prefix="", limit=200, client=None):
    """Overview for browsing: files per zone (title, description), optionally filtered."""
    lim = max(1, min(int(limit), 1000))
    clause, cparams = private_sql(sees_private(client))
    rows = db.execute(
        "SELECT d.path, d.zone, d.title, d.descr, (SELECT COUNT(*) FROM chunks c WHERE c.doc=d.id) "
        "FROM docs d WHERE (?='' OR d.zone=?) AND (?='' OR d.path LIKE ? ESCAPE '\\')" + clause +
        " ORDER BY d.prior DESC, d.path LIMIT ?",
        (zone, zone, prefix, prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", *cparams, lim)).fetchall()
    return {"zones": status(db)["zones"],
            "files": [{"path": r[0], "zone": r[1], "title": r[2], "descr": r[3], "sections": r[4]} for r in rows]}
