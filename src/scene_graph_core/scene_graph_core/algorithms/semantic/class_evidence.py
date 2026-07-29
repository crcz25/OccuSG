"""Accumulated per-class evidence and canonical class selection.

An object's canonical class is decided from every observation it has ever
absorbed, never from the newest detection alone. Evidence is a plain
``{class_name: weight}`` mapping so it stays JSON-serializable and framework-free.
"""

from math import isfinite
from typing import Any, Dict, Mapping, Optional, Tuple


def _clean_class_name(class_name: Any) -> Optional[str]:
    if class_name is None:
        return None
    label = " ".join(str(class_name).split())
    return label or None


def _clean_weight(weight: Any, default: float = 1.0) -> float:
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return float(default)
    if not isfinite(value) or value <= 0.0:
        return float(default)
    return value


def accumulate_class_evidence(
    evidence: Optional[Mapping[str, float]],
    class_name: Any,
    weight: Any = 1.0,
) -> Dict[str, float]:
    """Add one weighted observation to a class-evidence map.

    Unusable labels leave the evidence unchanged. Weights default to 1.0 when the
    supplied confidence is missing or not a positive finite number.
    """
    accumulated: Dict[str, float] = {}
    if isinstance(evidence, Mapping):
        for key, value in evidence.items():
            label = _clean_class_name(key)
            if label is None:
                continue
            try:
                current = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(current) and current > 0.0:
                accumulated[label] = current

    label = _clean_class_name(class_name)
    if label is None:
        return accumulated

    accumulated[label] = accumulated.get(label, 0.0) + _clean_weight(weight)
    return accumulated


def canonical_class_from_evidence(
    evidence: Optional[Mapping[str, float]],
) -> Tuple[Optional[str], float]:
    """Return ``(class_name, confidence)`` for the strongest accumulated class.

    Confidence is the winning class's share of the total evidence mass, so it is
    always in ``(0, 1]``. Ties break on the class name for determinism.
    """
    if not isinstance(evidence, Mapping) or not evidence:
        return None, 0.0

    usable: Dict[str, float] = {}
    for key, value in evidence.items():
        label = _clean_class_name(key)
        if label is None:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(weight) and weight > 0.0:
            usable[label] = weight

    if not usable:
        return None, 0.0

    total = sum(usable.values())
    winner = min(usable.items(), key=lambda item: (-item[1], item[0]))
    if total <= 0.0:
        return winner[0], 0.0
    return winner[0], float(winner[1] / total)
