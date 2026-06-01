Factories
- To decouple “what kind of node/edge do I want?” from “how do I actually build it?”
- If new types are needed, we just need to register one more function and the rest of the code still calls a single creat_edge() or create_node() method.
  - Edge Factory:
     - Allows edge-creation logic to be encapsulated in a single place.
     - Can have different factories for different edge types. and a single edge_factory.create_edge() method that can be used to create any edge type.
  - Node Factory (optional):
     - Did not considered needed as there seems to not be centralized constraints on nodes. (e.g., door must always have a pose in the room).
     - Similar to edge factory, but for nodes.
     - Can handle different node types and encapsulate their creation logic.

Observer:
  - Can be added to we listen for changes on the graph each time it is updated without having the manager to know about them.
  - the scene graph manager handles the graph behaviour, while the observer handles the graph changes
  - Allows to only manage those parts that are being processed.
  - If we expect to have multiple independent modules/processes to act on graph changes (and in real-time). we can use an observer to notify them of changes.

Scene Graph Manager
  - It is the centralized point of the "API"
  - store it as network.DiGraph
  - CRUD operations on the graph (add, remove, update nodes/edges)
  - we can implement high-level relationship generation methods (e.g., generate all doors in a room, or all rooms connected to a door)
  - We can expose specific tasks like cluster_rooms, find_nearest_room, find_doors etc. merge_all (for multiple robots)

Queries:
  - Create helper methods such as get_objects_in_room, get_all_nav_regions, etc.
  - methods to extract subgraphs, get_subgraphs_by_node_type, get_subgraphs_by_edge_type, etc.

Serialization:
  - Save to json, load from json, save to graphml, load from graphml.

ROS:
 - In practice ros2 should never touch the object directly, it only calls the methods in the graph manager to ros is only used to “where do I get the data?” rather than “how do I store and query a graph?”
 - it subsrices to the marker array, converts the markers into a scene node (extracts the type, pose, etc.) and then calls the graph manager to add the node to the graph.
 - if we detect the same object again, just update the existing node if it exists, otherwise create a new one (probably need to add a check for that based on the type of node, distance, etc. to avoid duplicates).
 - If the object is no longer detected, we can remove it from the graph (or mark it as inactive).

For room classification:
  - we can either let the ros node run the clustering routine and just call the graph manager to add based on the type of node, or we can have the graph manager handle the clustering and add the nodes directly.

So yes, the ROS2 node is “in charge” of deciding:
    - When a new object appears or disappears.
    -Which nodes should be connected by an edge, and what type of edge that is.
    - When to re‐run your clustering algorithm to update which room each object belongs to (or, alternatively, have the core library expose cluster_rooms(...) and let the ROS node invoke it whenever new data arrives).
  - ingest data, decide which relationshipts to create/demove in real time and call add, update, remove edges/nodes on the graph manager.



Note on “Tree Structure” vs. General Graph
    - Although you call it a “tree,” you actually have a forest of directed acyclic graphs (DAGs), where each layer enforces its own topology.
    - The global and semantic layers are tree‐like (“has_floor”, “has_room”, “contains” all flow downward).
    - The nav and motion layers can be more general graphs (bidirectional adjacency, loop closures, etc.).
    - Low‐level geometry might even form cycles if two mesh chunks overlap.


Notes on using networkx:
 - Can keep up if the graph is in the order of 10^4-10^5 nodes/edges.
 - Riuch built-in algorithms
 - not thread safe, so we need to handle concurrency ourselves (multiple robots updating the graph at the same time).
 - Start seeing problems with ~50k nodes/edges.
