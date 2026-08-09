"""Canonical runtime representation for OBJECT node attributes.

OBJECT nodes deliberately keep perception state in disjoint groups.  The
serializer moves these groups to the corresponding JSON node fields; it must
never emit both the grouped state and a second projection of it.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


OBJECT_ATTRIBUTE_GROUPS = (
    "geometry",
    "semantic",
    "detection",
    "embeddings",
    "observations",
)

OBJECT_SEMANTIC_KEYS = (
    "class_name",
)

OBJECT_DETECTION_KEYS = (
    "detection_confidence",
    "detector_source",
    "semantic_perception_class_id",
)

OBJECT_EMBEDDING_KEYS = (
    "object_embedding",
    "label_embedding",
    "mask_embedding",
    "bbox_embedding",
    "fused_embedding",
    "sum",
)

OBJECT_OBSERVATION_KEYS = (
    "detection_observation_count",
    "embedding_observation_count",
    "last_semantic_similarity",
)

OBJECT_GEOMETRY_KEYS = ("bbox_3d_size",)

OBJECT_KNOWN_KEYS = frozenset(
    {
        *OBJECT_SEMANTIC_KEYS,
        *OBJECT_DETECTION_KEYS,
        *OBJECT_EMBEDDING_KEYS,
        *OBJECT_OBSERVATION_KEYS,
        *OBJECT_GEOMETRY_KEYS,
    }
)


def empty_object_attributes() -> Dict[str, Any]:
    """Return an empty canonical attribute bag without empty substructures."""
    return {}


def object_group(
    attributes: Mapping[str, Any] | None,
    group: str,
    *,
    create: bool = False,
) -> Dict[str, Any]:
    """Return one canonical OBJECT group.

    Runtime writers use ``create=True`` before mutation.  Readers receive an
    empty mapping when the group is absent or malformed.
    """
    if not isinstance(attributes, dict):
        return {}
    value = attributes.get(group)
    if isinstance(value, dict):
        return value
    if create:
        value = {}
        attributes[group] = value
        return value
    return {}


def split_object_attributes(attributes: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    """Copy canonical groups for serialization, omitting empty groups."""
    if not isinstance(attributes, Mapping):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for group in OBJECT_ATTRIBUTE_GROUPS:
        value = attributes.get(group)
        if isinstance(value, Mapping) and value:
            result[group] = dict(value)
    return result


def validate_object_attributes(attributes: Mapping[str, Any] | None) -> None:
    """Validate the grouped runtime shape of OBJECT attributes."""
    if not isinstance(attributes, Mapping):
        return
    invalid_groups = sorted(
        key
        for key in attributes
        if key in OBJECT_ATTRIBUTE_GROUPS and not isinstance(attributes[key], dict)
    )
    if invalid_groups:
        raise ValueError(
            "OBJECT attribute groups must be dictionaries: "
            + ", ".join(invalid_groups)
        )
