"""Optional local embeddings: EmbeddingGemma-300m (4-bit ONNX graph, 768 dimensions) on CPU.

The model constants, prompts, batching and pooling follow `markdown-memory` 0.5.0 (MIT, Hesham Karm),
which the original service used as a dependency; see THIRD_PARTY_NOTICES.md. Only the parts needed here
are reimplemented on top of onnxruntime, tokenizers, numpy and huggingface_hub (extra `memidx[vectors]`).

Model files are never bundled and never downloaded implicitly: `memidx model download` fetches them.
The model has its own license terms (Gemma Terms of Use) which the user must accept.

Embedder hook
-------------
Any object with `embed_documents(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]`
can replace the built-in model:
  * in Python: `memidx.embed.set_embedder(obj)` (used by the tests), or
  * by configuration: `embedder = "package.module:factory"` / `MEMIDX_EMBEDDER=...`; `factory(config)`
    or `factory()` must return such an object.
An optional `name` attribute identifies the embedder in the index; when it changes, stored vectors are
discarded on the next `update --embed`.
"""
import hashlib
import importlib
import os
import sys
import threading
from pathlib import Path

REPOSITORY = "onnx-community/embeddinggemma-300m-ONNX"
REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"   # pinned: an upstream re-export must not change stored vectors
MODEL_FILE = "onnx/model_q4.onnx"
MANIFEST = {                                             # size and sha256 at REVISION
    MODEL_FILE: (519_322, "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043"),
    MODEL_FILE + "_data": (196_725_760, "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02"),
    "tokenizer.json": (20_323_312, "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
}
DIMENSION = 768
MAX_TOKENS = 512
QUERY_PROMPT = "task: search result | query: "          # prompts from the EmbeddingGemma model card
DOCUMENT_PROMPT = "title: none | text: "
BATCH_SIZE = 4
LICENSE_NOTICE = ("EmbeddingGemma-300m is distributed by Google under the Gemma Terms of Use "
                  "(https://ai.google.dev/gemma/terms). It is not covered by memidx's license. By downloading "
                  "and using the model you must accept those terms yourself.")

_EMB = None
_LOCK = threading.Lock()
_NOTICE_SHOWN = set()


class VectorsUnavailable(RuntimeError):
    """Vector search cannot run (missing extra, missing model files)."""


def notice(msg):
    """Print a one-line notice once per process."""
    if msg not in _NOTICE_SHOWN:
        _NOTICE_SHOWN.add(msg)
        print(f"memidx: {msg}", file=sys.stderr, flush=True)


def deps_available():
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def numpy_available():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def model_dir(cache_dir):
    return Path(cache_dir) / f"embeddinggemma-300m-onnx-{REVISION[:12]}"


def missing_files(cache_dir):
    """Model files that are absent or have the wrong size (cheap check; sha256 is checked after download)."""
    d = model_dir(cache_dir)
    out = []
    for name, (size, _) in MANIFEST.items():
        p = d / name
        if not p.is_file() or p.stat().st_size != size:
            out.append(name)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(cache_dir, log=print):
    """Fetch the pinned model files into `cache_dir` and verify their checksums. Network access."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise VectorsUnavailable("huggingface_hub is missing; install `memidx[vectors]`") from e
    d = model_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    log(LICENSE_NOTICE)
    log(f"Downloading {REPOSITORY}@{REVISION[:12]} (~218 MB) to {d}")
    # local_dir: real files, not symlinks into a blob store (onnxruntime refuses an external-data symlink)
    snapshot_download(REPOSITORY, revision=REVISION, allow_patterns=list(MANIFEST), local_dir=str(d))
    bad = [n for n, (size, sha) in MANIFEST.items()
           if not (d / n).is_file() or (d / n).stat().st_size != size or _sha256(d / n) != sha]
    if bad:
        raise VectorsUnavailable(f"downloaded files do not match the expected checksums: {', '.join(bad)}")
    log("Model files verified.")
    return d


class GemmaEmbedder:
    """EmbeddingGemma-300m via onnxruntime; loads lazily, never downloads."""

    name = f"{REPOSITORY}@{REVISION[:12]}/{MODEL_FILE}"

    def __init__(self, cache_dir):
        self.dir = model_dir(cache_dir)
        self._lock = threading.Lock()
        self._session = None
        self._tok = None

    def _load(self):
        with self._lock:
            if self._session is None:
                import onnxruntime
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
                tok.enable_truncation(max_length=MAX_TOKENS)
                tok.enable_padding()
                opts = onnxruntime.SessionOptions()
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                threads = os.environ.get("MEMIDX_THREADS", "").strip()
                if threads.isdigit() and int(threads) > 0:
                    opts.intra_op_num_threads = int(threads)
                self._session = onnxruntime.InferenceSession(str(self.dir / MODEL_FILE), opts,
                                                             providers=["CPUExecutionProvider"])
                self._tok = tok
            return self._session, self._tok

    def _embed(self, texts):
        if not texts:
            return []
        import numpy as np
        session, tok = self._load()
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths share a batch
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            enc = tok.encode_batch([texts[i] for i in batch])
            vecs = session.run(["sentence_embedding"], {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            })[0]
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=-1, keepdims=True), 1e-12)
            for i, v in zip(batch, vecs):
                out[i] = v.astype("float32")
        if any(len(v) != DIMENSION for v in out):
            raise RuntimeError(f"model did not return {DIMENSION}-dimensional vectors")
        return out

    def embed_documents(self, texts):
        return self._embed([DOCUMENT_PROMPT + t for t in texts])

    def embed_query(self, text):
        return self._embed([QUERY_PROMPT + text])[0]


def set_embedder(obj):
    """Install an embedder object (hook for tests and custom models); None resets to the default."""
    global _EMB
    with _LOCK:
        _EMB = obj


def embedder_name(emb):
    return getattr(emb, "name", None) or type(emb).__qualname__


def get_embedder(cfg):
    """Return the active embedder or raise VectorsUnavailable with a one-line reason."""
    global _EMB
    with _LOCK:
        if _EMB is not None:
            return _EMB
        if cfg.embedder:
            mod, _, attr = cfg.embedder.partition(":")
            factory = getattr(importlib.import_module(mod), attr or "create")
            try:
                _EMB = factory(cfg)
            except TypeError:
                _EMB = factory()
            return _EMB
        if not deps_available():
            raise VectorsUnavailable("vector extra not installed (pip install 'memidx[vectors]'); using full-text search only")
        if missing_files(cfg.model_cache):
            raise VectorsUnavailable(f"embedding model not found in {cfg.model_cache} (run `memidx model download`); "
                                     "using full-text search only")
        _EMB = GemmaEmbedder(cfg.model_cache)
        return _EMB
