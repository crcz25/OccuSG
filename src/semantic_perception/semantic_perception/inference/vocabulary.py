"""HM3D class vocabulary loading and Grounding DINO prompt construction.

The vocabulary is the single source of truth for object class names. Every entry
keeps a stable *global index*, and the prompt carries an explicit local-to-global
index map so a detector result can always be traced back to exactly one label.

All classes go into a single caption, ``"class1 . class2 . class3 ."``, so one
detector pass covers the whole detector vocabulary
(https://github.com/IDEA-Research/GroundingDINO/issues/85).

Labels can be excluded from that caption without touching the vocabulary itself,
so their cached text embeddings stay valid and the exclusion is reversible from
configuration alone.

This module has no ROS and no model dependencies.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence, Tuple


class VocabularyError(ValueError):
    """Raised when the class-label source cannot produce a usable vocabulary."""


def canonical_label(label: str) -> str:
    """Return the lookup key for a label: trimmed and case-folded.

    Only surrounding whitespace and case are normalized; the label text itself is
    never rewritten, so the published ``class_name`` keeps its original spelling.
    """
    return " ".join(str(label).split()).casefold()


def _iter_rows(text: str) -> Iterator[Tuple[int, list[str]]]:
    lines = text.splitlines()
    first_line = next((line for line in lines if line.strip()), "")
    # HM3D ships a "Object Type Name; # of instances" table; plain one-column
    # CSV vocabularies are also accepted.
    delimiter = ";" if first_line.casefold().startswith("object type name;") else ","
    for row_number, row in enumerate(csv.reader(lines, delimiter=delimiter), start=1):
        yield row_number, row


def load_class_labels(path: str | os.PathLike[str]) -> Tuple[list[str], bytes]:
    """Load class labels from a CSV, returning ``(labels, raw_bytes)``.

    Headers, empty rows, malformed entries, and duplicates are skipped. Ordering
    follows the file, which for the HM3D table means descending instance count.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise VocabularyError(f"Class label file does not exist: {source_path}")
    source = source_path.read_bytes()
    try:
        text = source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise VocabularyError(f"Class label file is not UTF-8: {source_path}") from exc

    labels: list[str] = []
    seen: set[str] = set()
    for row_number, row in _iter_rows(text):
        if not row or all(not value.strip() for value in row):
            continue
        label = " ".join(row[0].split())
        if not label:
            continue
        if row_number == 1 and "object type name" in label.casefold():
            continue
        key = canonical_label(label)
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)

    if not labels:
        raise VocabularyError(f"Class label file contains no valid labels: {source_path}")
    return labels, source


# Grounding DINO separates classes in a caption with " . " and terminates the
# caption with " .". Both are required for its text self-attention masks to split
# the caption into one span per class.
PROMPT_SEPARATOR = " . "
PROMPT_TERMINATOR = " ."


@dataclass(frozen=True)
class DetectorPrompt:
    """One Grounding DINO caption plus its local-to-global index map.

    ``global_indices[i]`` is the index of ``labels[i]`` in the *full* vocabulary.
    Excluding labels makes the prompt a non-contiguous subset, so this map is the
    only correct way back to a vocabulary entry or its cached text embedding.
    """

    labels: Tuple[str, ...]
    global_indices: Tuple[int, ...]
    excluded_labels: Tuple[str, ...] = ()

    @property
    def caption(self) -> str:
        """Return the single caption covering every label in this prompt."""
        if not self.labels:
            return ""
        return PROMPT_SEPARATOR.join(self.labels) + PROMPT_TERMINATOR

    def global_index(self, local_index: int) -> int | None:
        """Map a detector-local class index back to a global vocabulary index."""
        if not isinstance(local_index, (int,)) or isinstance(local_index, bool):
            return None
        if local_index < 0 or local_index >= len(self.global_indices):
            return None
        return int(self.global_indices[local_index])


def parse_excluded_labels(value: object) -> Tuple[str, ...]:
    """Parse a comma-separated exclusion list into ordered, unique labels.

    Accepts either a single ``"floor, ceiling"`` string or an already-split
    sequence. Blank entries are dropped and case/whitespace duplicates collapse.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        try:
            parts = [str(item) for item in value]
        except TypeError:
            return ()

    labels: list[str] = []
    seen: set[str] = set()
    for part in parts:
        label = " ".join(str(part).split())
        if not label:
            continue
        key = canonical_label(label)
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return tuple(labels)


@dataclass(frozen=True)
class Vocabulary:
    """An ordered, duplicate-free class vocabulary with canonical-name lookup."""

    labels: Tuple[str, ...]
    source_bytes: bytes = b""

    def __post_init__(self) -> None:
        if not self.labels:
            raise VocabularyError("Vocabulary must contain at least one label")
        lookup = {}
        for index, label in enumerate(self.labels):
            key = canonical_label(label)
            if key in lookup:
                raise VocabularyError(f"Vocabulary contains a duplicate label: {label!r}")
            lookup[key] = index
        object.__setattr__(self, "_lookup", lookup)

    @classmethod
    def from_csv(cls, path: str | os.PathLike[str]) -> "Vocabulary":
        labels, source = load_class_labels(path)
        return cls(tuple(labels), source)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def lookup(self) -> Mapping[str, int]:
        return dict(getattr(self, "_lookup"))

    def index_of(self, label: str) -> int | None:
        """Return the global index for ``label``, or ``None`` when unknown."""
        return getattr(self, "_lookup").get(canonical_label(label))

    def label_at(self, global_index: int) -> str | None:
        if 0 <= int(global_index) < len(self.labels):
            return self.labels[int(global_index)]
        return None

    def unknown_labels(self, labels: Sequence[str]) -> Tuple[str, ...]:
        """Return the supplied labels that this vocabulary does not contain."""
        return tuple(label for label in labels if self.index_of(label) is None)

    def build_prompt(
        self,
        excluded_labels: Sequence[str] = (),
        size: int = 0,
    ) -> DetectorPrompt:
        """Return the single detector caption for this vocabulary.

        Exclusions are applied first, then ``size`` keeps the leading labels of
        what remains, so excluding a frequent class backfills the caption with the
        next most frequent one instead of shrinking it. ``size <= 0`` keeps every
        remaining label. Excluded labels stay in the vocabulary and keep their
        cached text embeddings; they are only absent from the prompt.
        """
        excluded_keys = {canonical_label(label) for label in excluded_labels}
        selected = [
            (index, label)
            for index, label in enumerate(self.labels)
            if canonical_label(label) not in excluded_keys
        ]
        if size > 0:
            selected = selected[: int(size)]
        if not selected:
            raise VocabularyError(
                "Every vocabulary label is excluded; the detector prompt is empty"
            )
        return DetectorPrompt(
            labels=tuple(label for _, label in selected),
            global_indices=tuple(int(index) for index, _ in selected),
            excluded_labels=tuple(excluded_labels),
        )


def resolve_global_index(
    prompt: DetectorPrompt,
    vocabulary: Vocabulary,
    local_index: int | None,
    phrase: str | None,
) -> int | None:
    """Resolve one detector output to a global vocabulary index.

    The detector's local class index is authoritative. When it is missing (the
    phrase did not match any prompt exactly), the raw phrase is looked up in the
    vocabulary as a fallback. Unresolvable results return ``None`` so the caller
    can reject the detection rather than inventing a label.
    """
    if local_index is not None:
        resolved = prompt.global_index(int(local_index))
        if resolved is not None:
            return resolved
    if phrase:
        return vocabulary.index_of(phrase)
    return None


def build_detector_prompt(
    vocabulary: Vocabulary,
    detector_vocabulary_size: int,
    excluded_labels: Sequence[str] = (),
) -> DetectorPrompt:
    """Return the single detector prompt for one vocabulary and exclusion list."""
    return vocabulary.build_prompt(
        excluded_labels=excluded_labels,
        size=int(detector_vocabulary_size),
    )


def summarize(prompt: DetectorPrompt) -> str:
    """Return a one-line startup summary of the prompt configuration."""
    summary = (
        f"{len(prompt.labels)} detector labels in one caption "
        f"({len(prompt.caption)} characters)"
    )
    if prompt.excluded_labels:
        summary += f", excluding {list(prompt.excluded_labels)}"
    return summary
