"""Class-label CSV loading and a small, deterministic binary embedding cache."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

from semantic_perception.inference.vocabulary import (
    Vocabulary,
    VocabularyError,
    canonical_label,
    load_class_labels,
)

MAGIC = b"SPTE"
VERSION = 1
HEADER = struct.Struct("<4sIII32s")


class EmbeddingCacheError(ValueError):
    """Raised when prompt or cache data is malformed."""


def load_prompts(path: str | os.PathLike[str]) -> Tuple[list[str], bytes]:
    """Load class labels from a one-column CSV or an HM3D ``label; count`` table."""
    return load_class_labels(path)


def source_hash(csv_contents: bytes, model_name: str) -> bytes:
    """Hash the exact CSV bytes and model identifier without concatenation ambiguity."""
    model = model_name.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(csv_contents)))
    digest.update(csv_contents)
    digest.update(struct.pack("<Q", len(model)))
    digest.update(model)
    return digest.digest()


def read_cache(
    path: str | os.PathLike[str], expected_hash: bytes
) -> np.ndarray:
    """Read and strictly validate a cache file."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise EmbeddingCacheError("Embedding cache is truncated")
    magic, version, count, dimension, cached_hash = HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise EmbeddingCacheError("Embedding cache has invalid magic")
    if version != VERSION:
        raise EmbeddingCacheError(f"Unsupported embedding cache version: {version}")
    if cached_hash != expected_hash:
        raise EmbeddingCacheError("Embedding cache source hash does not match")
    if count == 0 or dimension == 0:
        raise EmbeddingCacheError("Embedding cache has invalid dimensions")
    expected_size = HEADER.size + count * dimension * np.dtype("<f4").itemsize
    if len(raw) != expected_size:
        raise EmbeddingCacheError("Embedding cache size does not match its header")
    matrix = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(count, dimension)
    if not np.isfinite(matrix).all():
        raise EmbeddingCacheError("Embedding cache contains non-finite values")
    return matrix.astype(np.float32, copy=True)


def write_cache(
    path: str | os.PathLike[str], matrix: np.ndarray, digest: bytes
) -> None:
    """Atomically write an embedding matrix using the documented wire format."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise EmbeddingCacheError("Text encoder returned an invalid embedding matrix")
    if len(digest) != 32:
        raise EmbeddingCacheError("Source hash must contain 32 bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = HEADER.pack(MAGIC, VERSION, values.shape[0], values.shape[1], digest)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(header)
            stream.write(values.astype("<f4", copy=False).tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_or_generate(
    csv_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    model_name: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    logger: Callable[[str], None] | None = None,
) -> tuple[list[str], np.ndarray, bool]:
    """Load a matching cache or regenerate it. Returns ``(..., cache_was_reused)``."""
    prompts, contents = load_prompts(csv_path)
    digest = source_hash(contents, model_name)
    try:
        matrix = read_cache(cache_path, digest)
        if matrix.shape[0] != len(prompts):
            raise EmbeddingCacheError("Embedding cache class count does not match CSV")
        if logger:
            logger(f"Reused text embedding cache: {cache_path}")
        return prompts, matrix, True
    except (FileNotFoundError, OSError, EmbeddingCacheError) as exc:
        if logger:
            logger(f"Regenerating text embedding cache ({exc})")

    matrix = np.asarray(encoder(prompts), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(prompts):
        raise EmbeddingCacheError(
            "Text encoder output must have shape [number_of_classes, embedding_dimension]"
        )
    matrix = _normalize_rows(matrix)
    write_cache(cache_path, matrix, digest)
    if logger:
        logger(
            f"Generated text embedding cache: {cache_path} "
            f"({matrix.shape[0]} prompts x {matrix.shape[1]} dimensions)"
        )
    return prompts, matrix, False


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise EmbeddingCacheError("Text encoder returned zero or non-finite embeddings")
    return (matrix / norms).astype(np.float32)


@dataclass(frozen=True)
class LabelEmbeddingCache:
    """Immutable CLIP text embeddings for one vocabulary, keyed by class index.

    Every row is L2-normalized and finite. Lookups accept a global vocabulary
    index or a canonical class name; a miss returns ``None`` so the caller can
    reject the detection instead of substituting a zero vector.
    """

    vocabulary: Vocabulary
    embeddings: np.ndarray
    dimension: int
    encode_seconds: float
    reused_cache: bool

    @property
    def nbytes(self) -> int:
        return int(self.embeddings.nbytes)

    def by_index(self, global_index: int | None) -> np.ndarray | None:
        if global_index is None:
            return None
        index = int(global_index)
        if index < 0 or index >= self.embeddings.shape[0]:
            return None
        return self.embeddings[index]

    def by_name(self, class_name: str | None) -> np.ndarray | None:
        if not class_name:
            return None
        return self.by_index(self.vocabulary.index_of(class_name))

    def classify(self, image_embedding: np.ndarray) -> Tuple[int, float] | None:
        """Return the closest label as ``(global_index, cosine_similarity)``.

        The image embedding must already be unit-norm and of the cache dimension.
        Used only for debug-image annotation; no pipeline decision depends on it.
        """
        value = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
        if value.size != self.dimension or not np.isfinite(value).all():
            return None
        scores = self.embeddings @ value
        if scores.size == 0 or not np.isfinite(scores).all():
            return None
        best = int(np.argmax(scores))
        return best, float(scores[best])

    def summary(self) -> str:
        return (
            f"Encoded {self.embeddings.shape[0]} class labels "
            f"({self.dimension}-D, {self.nbytes / 1024.0:.1f} KiB) in "
            f"{self.encode_seconds:.2f} s "
            f"({'reused cache' if self.reused_cache else 'generated cache'})"
        )


def build_label_embedding_cache(
    csv_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    model_name: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    logger: Callable[[str], None] | None = None,
) -> LabelEmbeddingCache:
    """Load the vocabulary and its CLIP text embeddings exactly once."""
    started = time.monotonic()
    labels, matrix, reused = load_or_generate(
        csv_path, cache_path, model_name, encoder, logger
    )
    elapsed = time.monotonic() - started
    vocabulary = Vocabulary(tuple(labels))
    values = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if values.ndim != 2 or values.shape[0] != len(vocabulary) or values.shape[1] == 0:
        raise EmbeddingCacheError(
            "Label embedding matrix must have shape [number_of_labels, dimension]"
        )
    if not np.isfinite(values).all():
        raise EmbeddingCacheError("Label embedding cache contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise EmbeddingCacheError("Label embedding cache contains a zero-norm entry")
    values = (values / norms[:, None]).astype(np.float32)
    values.setflags(write=False)
    return LabelEmbeddingCache(
        vocabulary=vocabulary,
        embeddings=values,
        dimension=int(values.shape[1]),
        encode_seconds=float(elapsed),
        reused_cache=bool(reused),
    )


__all__ = [
    "EmbeddingCacheError",
    "LabelEmbeddingCache",
    "Vocabulary",
    "VocabularyError",
    "build_label_embedding_cache",
    "canonical_label",
    "load_or_generate",
    "load_prompts",
    "read_cache",
    "source_hash",
    "write_cache",
]
