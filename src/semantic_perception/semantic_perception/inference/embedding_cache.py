"""Prompt CSV loading and a small, deterministic binary embedding cache."""

from __future__ import annotations

import csv
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

MAGIC = b"SPTE"
VERSION = 1
HEADER = struct.Struct("<4sIII32s")


class EmbeddingCacheError(ValueError):
    """Raised when prompt or cache data is malformed."""


def load_prompts(path: str | os.PathLike[str]) -> Tuple[list[str], bytes]:
    """Load a one-column CSV or an HM3D ``label; count`` table."""
    source = Path(path).read_bytes()
    try:
        text = source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EmbeddingCacheError(f"Prompt CSV is not UTF-8: {path}") from exc

    lines = text.splitlines()
    first_line = next((line for line in lines if line.strip()), "")
    hm3d_counts = first_line.casefold().startswith("object type name;")
    reader = csv.reader(lines, delimiter=";" if hm3d_counts else ",")
    prompts: list[str] = []
    for row_number, row in enumerate(reader, start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if hm3d_counts and row_number == 1:
            if len(row) != 2 or "object type name" not in row[0].casefold():
                raise EmbeddingCacheError("HM3D CSV has an invalid header")
            continue
        if hm3d_counts:
            if len(row) != 2 or not row[0].strip():
                raise EmbeddingCacheError(
                    f"HM3D CSV row {row_number} must contain a class and count"
                )
            try:
                count = int(row[1].strip())
            except ValueError as exc:
                raise EmbeddingCacheError(
                    f"HM3D CSV row {row_number} has an invalid instance count"
                ) from exc
            if count < 0:
                raise EmbeddingCacheError(
                    f"HM3D CSV row {row_number} has a negative instance count"
                )
            prompts.append(row[0].strip())
            continue
        if len(row) != 1 or not row[0].strip():
            raise EmbeddingCacheError(
                f"Prompt CSV row {row_number} must contain exactly one class name"
            )
        prompts.append(row[0].strip())

    if not prompts:
        raise EmbeddingCacheError(f"Prompt CSV contains no classes: {path}")
    if len(set(prompts)) != len(prompts):
        raise EmbeddingCacheError("Prompt CSV contains duplicate class names")
    return prompts, source


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
    return prompts, matrix, False


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise EmbeddingCacheError("Text encoder returned zero or non-finite embeddings")
    return (matrix / norms).astype(np.float32)
