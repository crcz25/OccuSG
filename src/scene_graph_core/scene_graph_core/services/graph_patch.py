"""
GraphPatch - Atomic batch updates for the scene graph.

This module provides a way to describe a set of mutations (nodes and edges to add/remove,
attributes to update) that can be applied atomically. This is more efficient than
individual updates and allows for validation before applying changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..representation import BaseNode, Edge, EdgeType


@dataclass(frozen=True)
class EdgePatchAdd:
    """One edge add operation with explicit structural semantics."""

    edge: Edge
    is_structural: bool


@dataclass(frozen=True)
class EdgePatchRemove:
    """One edge remove operation with an optional type filter."""

    source_id: int
    target_id: int
    edge_type: Optional[EdgeType] = None


@dataclass
class GraphPatch:
    """
    A batch of graph mutations to be applied atomically.

    Attributes:
        nodes_to_add: List of nodes to add
        nodes_to_remove: Set of node IDs to remove
        nodes_to_update: Dict mapping node_id to replacement node
        edges_to_add: List of edge add operations
        edges_to_remove: List of edge remove operations
        node_attribute_updates: Dict mapping node_id to attribute updates
    """

    nodes_to_add: List[BaseNode] = field(default_factory=list)
    nodes_to_remove: Set[int] = field(default_factory=set)
    nodes_to_update: Dict[int, BaseNode] = field(default_factory=dict)
    edges_to_add: List[EdgePatchAdd] = field(default_factory=list)
    edges_to_remove: List[EdgePatchRemove] = field(default_factory=list)
    node_attribute_updates: Dict[int, Dict[str, any]] = field(default_factory=dict)

    def add_node(self, node: BaseNode) -> "GraphPatch":
        """Add one node to the patch."""
        self.nodes_to_add.append(node)
        return self

    def remove_node(self, node_id: int) -> "GraphPatch":
        """Mark one node for removal."""
        self.nodes_to_remove.add(int(node_id))
        return self

    def update_node(self, node_id: int, node: BaseNode) -> "GraphPatch":
        """Mark one node for full replacement."""
        self.nodes_to_update[int(node_id)] = node
        return self

    def add_edge(
        self,
        edge: Edge,
        *,
        is_structural: Optional[bool] = None,
    ) -> "GraphPatch":
        """Add one edge to the patch with explicit structural intent."""
        resolved_is_structural = (
            edge.is_structural if is_structural is None else bool(is_structural)
        )
        self.edges_to_add.append(
            EdgePatchAdd(edge=edge, is_structural=resolved_is_structural)
        )
        return self

    def remove_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: Optional[EdgeType] = None,
    ) -> "GraphPatch":
        """Mark one edge for removal."""
        self.edges_to_remove.append(
            EdgePatchRemove(
                source_id=int(source_id),
                target_id=int(target_id),
                edge_type=edge_type,
            )
        )
        return self

    def update_node_attributes(
        self, node_id: int, attributes: Dict[str, any]
    ) -> "GraphPatch":
        """Mark one node's attributes for update/merge."""
        node_id = int(node_id)
        if node_id not in self.node_attribute_updates:
            self.node_attribute_updates[node_id] = {}
        self.node_attribute_updates[node_id].update(attributes)
        return self

    def is_empty(self) -> bool:
        """Return True when the patch has no operations."""
        return not (
            self.nodes_to_add
            or self.nodes_to_remove
            or self.nodes_to_update
            or self.edges_to_add
            or self.edges_to_remove
            or self.node_attribute_updates
        )

    def size(self) -> int:
        """Return the total number of queued operations."""
        return (
            len(self.nodes_to_add)
            + len(self.nodes_to_remove)
            + len(self.nodes_to_update)
            + len(self.edges_to_add)
            + len(self.edges_to_remove)
            + len(self.node_attribute_updates)
        )

    def validate(self, graph) -> List[str]:
        """Validate the patch against a graph."""
        errors = []
        nodes_being_added = {n.id for n in self.nodes_to_add if n.id is not None}

        for node_id in self.nodes_to_remove:
            if not graph.has_node(node_id):
                errors.append(f"Cannot remove non-existent node {node_id}")

        for node_id in self.nodes_to_update:
            if node_id in self.nodes_to_remove:
                errors.append(f"Cannot update and remove node {node_id} in one patch")
            elif node_id not in nodes_being_added and not graph.has_node(node_id):
                errors.append(f"Cannot update non-existent node {node_id}")

        for edge_item in self.edges_to_add:
            edge = edge_item.edge
            if edge.source_id not in nodes_being_added and not graph.has_node(
                edge.source_id
            ):
                errors.append(
                    f"Cannot add edge: source node {edge.source_id} does not exist"
                )
            if edge.target_id not in nodes_being_added and not graph.has_node(
                edge.target_id
            ):
                errors.append(
                    f"Cannot add edge: target node {edge.target_id} does not exist"
                )

        for remove_op in self.edges_to_remove:
            if not graph.has_edge(
                remove_op.source_id,
                remove_op.target_id,
                edge_type=remove_op.edge_type,
            ):
                errors.append(
                    "Cannot remove non-existent edge "
                    f"{remove_op.source_id} -> {remove_op.target_id}"
                )

        for node_id in self.node_attribute_updates:
            if node_id not in nodes_being_added and not graph.has_node(node_id):
                errors.append(
                    f"Cannot update attributes of non-existent node {node_id}"
                )

        return errors

    def __str__(self) -> str:
        """String representation for debugging."""
        parts = []
        if self.nodes_to_add:
            parts.append(f"+{len(self.nodes_to_add)} nodes")
        if self.nodes_to_remove:
            parts.append(f"-{len(self.nodes_to_remove)} nodes")
        if self.nodes_to_update:
            parts.append(f"~{len(self.nodes_to_update)} nodes")
        if self.edges_to_add:
            parts.append(f"+{len(self.edges_to_add)} edges")
        if self.edges_to_remove:
            parts.append(f"-{len(self.edges_to_remove)} edges")
        if self.node_attribute_updates:
            parts.append(f"~{len(self.node_attribute_updates)} attrs")
        return f"GraphPatch({', '.join(parts) if parts else 'empty'})"

