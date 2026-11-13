"""Optional local embeddings. Vectors are opt-in (`vectors = true`) and the model is an explicit choice.

memidx ships no model. A model is a named preset (PRESETS) with a pinned revision, file checksums, prompts and
its license. `memidx model list` shows them; `memidx model download <preset>` fetches one (a preset whose
license is not OSI-approved needs `--accept-license`). Nothing is ever downloaded implicitly.

The only preset today, `embeddinggemma-300m` (4-bit ONNX graph, 768 dimensions, CPU), is distributed under the
Gemma Terms of Use, which is NOT an OSI-approved license. Its constants, prompts, batching and pooling follow
`markdown-memory` 0.5.0 (MIT, Hesham Karm); see THIRD_PARTY_NOTICES.md. Only the needed parts are reimplemented
on top of onnxruntime, tokenizers, numpy and huggingface_hub (extra `memidx[vectors]`).

Embedder hook
-------------
Any object with `embed_documents(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]`
can replace the preset model:
  * in Python: `memidx.embed.set_embedder(obj)` (used by the tests), or
  * by trusted configuration only: `embedder = "package.module:factory"` / `MEMIDX_EMBEDDER=...`;
    `factory(config)` or `factory()` must return such an object. This imports and runs that code.
An optional `name` attribute identifies the embedder in the index; when it changes, stored vectors are
discarded on the next `update`.
"""
import hashlib
import importlib
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelPreset:
    """Everything memidx needs to know about one downloadable embedding model. Add a preset by adding an entry
    to PRESETS; call sites only use the preset object."""
    name: str
    repository: str
    revision: str                       # pinned: an upstream re-export must not change stored vectors
    model_file: str
    files: dict = field(default_factory=dict)   # {relative path: (size in bytes, sha256)} at `revision`
    query_prompt: str = ""
    document_prompt: str = ""
    dimension: int = 0
    max_tokens: int = 512
    license_name: str = ""
    license_url: str = ""
    osi_approved: bool = False
    download_size: str = ""
    dir_name: str = ""                  # directory below the model cache

    @property
    def embedder_name(self):
        """Stored in the index (meta 'embedder'); a change discards stored vectors."""
        return f"{self.repository}@{self.revision[:12]}/{self.model_file}"

    @property
    def license_notice(self):
        osi = "an OSI-approved open-source license" if self.osi_approved else "NOT an OSI-approved open-source license"
        return (f"{self.name} ({self.repository}) is distributed under {self.license_name} ({self.license_url}), "
                f"which is {osi}. It is not covered by memidx's license; by downloading and using the model you "
                "accept those terms yourself.")


_GEMMA_FILE = "onnx/model_q4.onnx"
PRESETS = {
    "embeddinggemma-300m": ModelPreset(
        name="embeddinggemma-300m",
        repository="onnx-community/embeddinggemma-300m-ONNX",
        revision="5090578d9565bb06545b4552f76e6bc2c93e4a66",
        model_file=_GEMMA_FILE,
        files={
            _GEMMA_FILE: (519_322, "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043"),
            _GEMMA_FILE + "_data": (196_725_760, "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02"),
            "tokenizer.json": (20_323_312, "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
        },
        query_prompt="task: search result | query: ",      # prompts from the EmbeddingGemma model card
        document_prompt="title: none | text: ",
        dimension=768,
        max_tokens=512,
        license_name="the Gemma Terms of Use",
        license_url="https://ai.google.dev/gemma/terms",
        osi_approved=False,
        download_size="~218 MB",
        dir_name="embeddinggemma-300m-onnx-5090578d9565",
    ),
}

# 0.1.0 names, kept for scripts that imported them (values of the only preset)
_P = PRESETS["embeddinggemma-300m"]
REPOSITORY, REVISION, MODEL_FILE, MANIFEST = _P.repository, _P.revision, _P.model_file, _P.files
DIMENSION, MAX_TOKENS, QUERY_PROMPT, DOCUMENT_PROMPT = _P.dimension, _P.max_tokens, _P.query_prompt, _P.document_prompt
LICENSE_NOTICE = _P.license_notice
BATCH_SIZE = 4


def get_preset(name):
    p = PRESETS.get((name or "").strip())
    if p is None:
        raise ValueError(f"unknown model preset {name!r}; available: {', '.join(sorted(PRESETS))}")
    return p


_EMB = None
_LOCK = threading.Lock()
_NOTICE_SHOWN = set()


class VectorsUnavailable(RuntimeError):
    """Vector search cannot run (missing extra, missing model files): memidx falls back to full text."""


class ModelNotConfigured(ValueError):
    """vectors = true without a model preset or embedder hook: a configuration error (exit 2)."""


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


def model_dir(cache_dir, preset):
    return Path(cache_dir) / preset.dir_name


def missing_files(cache_dir, preset):
    """Model files that are absent or have the wrong size (cheap check; sha256 is checked after download)."""
    d = model_dir(cache_dir, preset)
    out = []
    for name, (size, _) in preset.files.items():
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


def download_model(cache_dir, preset, log=print):
    """Fetch the pinned files of `preset` into `cache_dir` and verify their checksums. Network access.
    The caller is responsible for the license confirmation (see cli.cmd_model)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise VectorsUnavailable("huggingface_hub is missing; install `memidx[vectors]`") from e
    d = model_dir(cache_dir, preset)
    d.mkdir(parents=True, exist_ok=True)
    log(preset.license_notice)
    log(f"Downloading {preset.repository}@{preset.revision[:12]} ({preset.download_size}) to {d}")
    # local_dir: real files, not symlinks into a blob store (onnxruntime refuses an external-data symlink)
    snapshot_download(preset.repository, revision=preset.revision, allow_patterns=list(preset.files), local_dir=str(d))
    bad = [n for n, (size, sha) in preset.files.items()
           if not (d / n).is_file() or (d / n).stat().st_size != size or _sha256(d / n) != sha]
    if bad:
        raise VectorsUnavailable(f"downloaded files do not match the expected checksums: {', '.join(bad)}")
    log("Model files verified.")
    return d


class OnnxEmbedder:
    """A preset model via onnxruntime (sentence_embedding output); loads lazily, never downloads."""

    def __init__(self, preset, cache_dir, threads=0):
        self.preset = preset
        self.name = preset.embedder_name
        self.dir = model_dir(cache_dir, preset)
        self.threads = int(threads or 0)
        self._lock = threading.Lock()
        self._session = None
        self._tok = None

    def _load(self):
        with self._lock:
            if self._session is None:
                import onnxruntime
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
                tok.enable_truncation(max_length=self.preset.max_tokens)
                tok.enable_padding()
                opts = onnxruntime.SessionOptions()
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                if self.threads > 0:
                    opts.intra_op_num_threads = self.threads
                self._session = onnxruntime.InferenceSession(str(self.dir / self.preset.model_file), opts,
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
        dim = self.preset.dimension
        if any(len(v) != dim for v in out):
            raise RuntimeError(f"model did not return {dim}-dimensional vectors")
        return out

    def embed_documents(self, texts):
        return self._embed([self.preset.document_prompt + t for t in texts])

    def embed_query(self, text):
        return self._embed([self.preset.query_prompt + text])[0]


GemmaEmbedder = OnnxEmbedder     # 0.1.0 name


def set_embedder(obj):
    """Install an embedder object (hook for tests and custom models); None resets to the default."""
    global _EMB
    with _LOCK:
        _EMB = obj


def current_embedder():
    return _EMB


def embedder_name(emb):
    return getattr(emb, "name", None) or type(emb).__qualname__


def get_embedder(cfg):
    """Return the active embedder or raise VectorsUnavailable (fall back to full text) / ModelNotConfigured."""
    global _EMB
    with _LOCK:
        if _EMB is not None:
            return _EMB
        if cfg.embedder:            # only from trusted sources (see config.CORPUS_KEYS)
            mod, _, attr = cfg.embedder.partition(":")
            factory = getattr(importlib.import_module(mod), attr or "create")
            try:
                _EMB = factory(cfg)
            except TypeError:
                _EMB = factory()
            return _EMB
        if not cfg.model:
            raise ModelNotConfigured("vectors are enabled but no model is configured; set `model = \"<preset>\"` "
                                     "(see `memidx model list`) or vectors = false")
        preset = get_preset(cfg.model)
        if not deps_available():
            raise VectorsUnavailable("vector extra not installed (pip install 'memidx[vectors]'); using full-text search only")
        if missing_files(cfg.model_cache, preset):
            raise VectorsUnavailable(f"embedding model not found in {cfg.model_cache} (run `memidx model download "
                                     f"{preset.name}`); using full-text search only")
        _EMB = OnnxEmbedder(preset, cfg.model_cache, getattr(cfg, "threads", 0))
        return _EMB
