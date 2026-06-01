"""Shared helpers for semantic signatures and weighted overlap scoring."""

from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, Mapping, Optional, Set, TypeVar

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import BaseNode, NodeType

SemanticToken = TypeVar("SemanticToken", bound=Hashable)
SemanticObjectTuple = tuple[str, float, float]


def get_object_class_name(object_node: Optional[BaseNode]) -> Optional[str]:
    """Return the canonical OBJECT class label from ``attributes['class_name']``."""
    if object_node is None or object_node.node_type != NodeType.OBJECT:
        return None

    attrs = object_node.attributes or {}
    class_name = attrs.get("class_name")
    if class_name is None:
        return None

    label = str(class_name).strip()
    if not label:
        return None

    return label


def normalize_signature_round_decimals(
    round_decimals: object,
    *,
    default: int = 1,
) -> int:
    """Return an integer ``round()`` precision for semantic object tuples."""
    try:
        return int(round_decimals)
    except (TypeError, ValueError):
        return int(default)


def build_object_signature_token(
    object_node: Optional[BaseNode],
    *,
    round_decimals: int = 1,
) -> Optional[SemanticObjectTuple]:
    """Return one comparable ``(class_name, x, y)`` token for an OBJECT node."""
    if object_node is None or object_node.node_type != NodeType.OBJECT:
        return None

    class_name = get_object_class_name(object_node)
    if class_name is None:
        return None

    position = getattr(getattr(object_node, "pose", None), "position", None)
    if position is None:
        return None

    try:
        x = float(position.x)
        y = float(position.y)
    except (AttributeError, TypeError, ValueError):
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None

    decimals = normalize_signature_round_decimals(round_decimals)
    return (class_name, round(x, decimals), round(y, decimals))


def serialize_object_signature_set(
    signature_set: Iterable[SemanticObjectTuple],
) -> list[dict[str, float | str]]:
    """Serialize tuple signatures into a stable JSON/log-friendly form."""
    serialized: list[dict[str, float | str]] = []
    for class_name, x, y in sorted(signature_set):
        serialized.append(
            {
                "class_name": str(class_name),
                "x": float(x),
                "y": float(y),
            }
        )
    return serialized


def build_object_signature_set_from_object_ids(
    sg_interface: SceneGraphInterface,
    object_ids: Iterable[int],
    *,
    round_decimals: int = 1,
) -> Set[SemanticObjectTuple]:
    """Build a semantic tuple-signature set from a unique set of OBJECT node IDs."""
    unique_object_ids = {
        int(object_id) for object_id in object_ids if object_id is not None
    }

    signature_set: Set[SemanticObjectTuple] = set()
    for object_id in unique_object_ids:
        object_node = sg_interface.query.get_node(int(object_id))
        signature_token = build_object_signature_token(
            object_node,
            round_decimals=round_decimals,
        )
        if signature_token is None:
            continue
        signature_set.add(signature_token)

    return signature_set


def build_class_set_from_object_ids(
    sg_interface: SceneGraphInterface,
    object_ids: Iterable[int],
) -> Set[str]:
    """Build a semantic class set from a unique set of OBJECT node IDs."""
    unique_object_ids = {
        int(object_id) for object_id in object_ids if object_id is not None
    }

    class_set: Set[str] = set()
    for object_id in unique_object_ids:
        object_node = sg_interface.query.get_node(int(object_id))
        class_name = get_object_class_name(object_node)
        if class_name is None:
            continue
        class_set.add(class_name)

    return class_set


def compute_inverse_document_frequency(
    signature_sets: Mapping[int, Iterable[SemanticToken]],
    *,
    extra_tokens: Iterable[SemanticToken] = (),
) -> Dict[SemanticToken, float]:
    """Compute corpus IDF weights across room signature sets."""
    room_ids = sorted(int(room_id) for room_id in signature_sets)
    if not room_ids:
        return {}

    document_frequency: Dict[SemanticToken, int] = {}
    for room_id in room_ids:
        for token in {token for token in signature_sets[int(room_id)] if token is not None}:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    all_tokens = set(document_frequency)
    all_tokens.update(token for token in extra_tokens if token is not None)

    room_count = float(len(room_ids))
    return {
        token: math.log(
            (1.0 + room_count) / (1.0 + float(document_frequency.get(token, 0)))
        )
        + 1.0
        for token in sorted(all_tokens)
    }


def weighted_tversky_set_similarity(
    query_signature_set: Iterable[SemanticToken],
    room_signature_set: Iterable[SemanticToken],
    token_weights: Mapping[SemanticToken, float],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute weighted Tversky similarity on semantic signature sets."""
    query_tokens = {token for token in query_signature_set if token is not None}
    room_tokens = {token for token in room_signature_set if token is not None}
    if not query_tokens and not room_tokens:
        return 0.0

    intersection_mass = sum(
        float(token_weights.get(token, 0.0))
        for token in sorted(query_tokens.intersection(room_tokens))
    )
    query_only_mass = sum(
        float(token_weights.get(token, 0.0))
        for token in sorted(query_tokens.difference(room_tokens))
    )
    room_only_mass = sum(
        float(token_weights.get(token, 0.0))
        for token in sorted(room_tokens.difference(query_tokens))
    )

    denominator = intersection_mass + alpha * query_only_mass + beta * room_only_mass
    if denominator <= 0.0:
        return 0.0

    return intersection_mass / denominator
