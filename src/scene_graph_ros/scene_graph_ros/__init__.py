"""
ROS 2 integration for the scene graph pipeline.

Main runtime components:

- SceneGraphOrchestrator (`scene_graph_region`): ingests odometry, detections,
  stable regions, and occupancy maps; promotes visited stable regions into the
  shared scene graph.
- VisualizationNode (`visualization_node`): publishes RViz markers from the
  shared scene graph interface.
- Utility nodes: `robot_trajectory_node`.

Main entry point:
    ros2 run scene_graph_ros scene_graph_region
"""

__version__ = "0.1.0"
