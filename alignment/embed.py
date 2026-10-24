"""Embed content chunks with a dense sentence-transformers model or a TF-IDF baseline.

Dense embeddings are cached in ``<cache>/<slug>/<sha256>.npy`` (+ ``.ids.json``). The hash
covers the model name, the prefix and the ordered ``(chunk_id, text)`` list, so any change
to the text forces a re-embed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def pick_device(device: str = "auto") -> str:
    """``cuda`` if available, then ``mps``, then ``cpu`` (unless a device is given)."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_key(model_name: str, prefix: str, items: list[tuple[str, str]]) -> str:
    """sha256 over the model, the prefix and the ordered ``(chunk_id, text)`` pairs."""
    payload = json.dumps(
        {"model": model_name, "prefix": prefix, "items": [list(i) for i in items]},
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def l2_normalize(E: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / np.where(norms == 0, 1, norms)).astype(np.float32)


def embed_tfidf(texts: list[str]) -> np.ndarray:
    """TF-IDF vectors fit on ``texts`` (all content chunks of both boards), L2-normalised."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    return l2_normalize(vectorizer.fit_transform(texts).toarray())


def embed_dense(
    model_name: str, texts: list[str], prefix: str, batch_size: int, device: str
) -> np.ndarray:
    """Unit-length sentence-transformers embeddings of ``prefix + text``."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    E = model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(E, dtype=np.float32)


def embed(chunks: list[dict], model_cfg: dict, emb_cfg: dict, cache_root: Path) -> np.ndarray:
    """One embedding row per chunk, in ``chunks`` order. Dense models go through the cache;
    TF-IDF is fast and deterministic, so it is recomputed every time."""
    texts = [c["text"] for c in chunks]
    if model_cfg["name"] == "tfidf":
        return embed_tfidf(texts)
    ids = [c["chunk_id"] for c in chunks]
    prefix = model_cfg.get("prefix", "")
    key = cache_key(model_cfg["name"], prefix, list(zip(ids, texts, strict=True)))
    folder = cache_root / model_cfg["slug"]
    npy, ids_json = folder / f"{key}.npy", folder / f"{key}.ids.json"
    if npy.exists() and ids_json.exists():
        if json.loads(ids_json.read_text(encoding="utf-8")) == ids:
            print(f"embeddings from cache: {npy}", flush=True)
            return np.load(npy)
    device = pick_device(emb_cfg.get("device", "auto"))
    print(f"embedding {len(texts)} chunks with {model_cfg['name']} on {device}", flush=True)
    E = embed_dense(model_cfg["name"], texts, prefix, emb_cfg["batch_size"], device)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(npy, E)
    ids_json.write_text(json.dumps(ids), encoding="utf-8")
    return E
