#!/usr/bin/env python3
"""
Launch file for point cloud generation with occupancy mapping pipeline.

This launch file orchestrates the complete pipeline from depth images to
occupancy grid maps:
  1. point_cloud_generator_node: Converts depth images to point clouds
  2. octomap_server: Builds 3D occupancy representation from point clouds
  3. map_conversion_node: Converts octomap to 2D occupancy grid

The nodes are started sequentially to ensure proper initialization order.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the complete mapping pipeline."""
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value="",
        description=(
            "Path to config YAML file (optional). "
            "If not provided, uses inline parameters."
        ),
    )

    # Try to load config from scene_graph_ros if available
    try:
        scene_graph_share = get_package_share_directory("scene_graph_ros")
        default_param_file = os.path.join(
            scene_graph_share, "config", "scene_graph_pipeline_params.yaml"
        )

        if os.path.exists(default_param_file):
            with open(default_param_file, "r") as f:
                all_params = yaml.safe_load(f)
                p_os = all_params.get("octomap_server", {}).get(
                    "ros__parameters", {}
                )
                p_mc = all_params.get("map_conversion_node", {}).get(
                    "ros__parameters", {}
                )
                p_pcg = all_params.get("point_cloud_generator_node", {}).get(
                    "ros__parameters", {}
                )
        else:
            # Fallback to inline parameters
            p_pcg = {}
            p_os = {}
            p_mc = {}
    except Exception:
        # Fallback to inline parameters if scene_graph_ros is not available
        p_pcg = {}
        p_os = {}
        p_mc = {}

    # Set default parameters if not loaded from config
    if not p_pcg:
        p_pcg = {
            "depth_image_topic": "/intel_realsense_r200_depth/depth/image_raw",
            "camera_info_topic": (
                "/intel_realsense_r200_depth/depth/camera_info"
            ),
            "output_frame": "camera_depth_optical_frame",
            "pointcloud_topic": "/pointcloud",
            "stride": 2,
            "min_depth": 0.01,
            "max_depth": 10.0,
            "target_hz": 0.0,
            "sync_queue_size": 10,
            "sync_slop_sec": 0.05,
            "qos_depth": 10,
            "input_qos_reliability": "best_effort",
            "output_qos_reliability": "reliable",
        }

    if not p_os:
        p_os = {
            "frame_id": "odom",
            "resolution": 0.1,
        }

    if not p_mc:
        p_mc = {
            "map_frame": "odom",
            "minimum_z": 1.0,
            "max_slope_ugv": 0.2,
            "slope_estimation_size": 2,
            "minimum_occupancy": 5,
            "partial_map_updates": True,
            "local_width": 10.0,
            "local_height": 10.0,
            "robot_frame": "base_link",
            "pose_source": "odom",
            "odom_topic": "/odom",
            "subscriber_qos_reliable": True,
            "subscriber_qos_transient_local": False,
            "publisher_qos_reliable": True,
            "publisher_qos_transient_local": False,
        }

    # Node 1: Point Cloud Generator
    point_cloud_generator = Node(
        package="point_cloud_generator",
        executable="depth_to_pointcloud",
        name="point_cloud_generator_node",
        output="screen",
        parameters=[p_pcg],
    )

    # Node 2: Octomap Server
    octomap_server = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server",
        output="screen",
        parameters=[p_os],
        remappings=[("cloud_in", "/pointcloud")],
    )

    # Node 3: Map Conversion
    map_conversion = Node(
        package="mapconversion",
        executable="map_conversion_oct_node",
        name="map_conversion_node",
        output="screen",
        parameters=[p_mc],
        remappings=[("octomap", "octomap_full")],
    )

    # Build launch description with sequential startup
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(config_file_arg)

    # 1) Start point_cloud_generator first
    ld.add_action(point_cloud_generator)

    # 2) When point_cloud_generator is up, start octomap_server
    ld.add_action(
        RegisterEventHandler(
            OnProcessStart(
                target_action=point_cloud_generator, on_start=[octomap_server]
            )
        )
    )

    # 3) When octomap_server is up, start map_conversion
    ld.add_action(
        RegisterEventHandler(
            OnProcessStart(
                target_action=octomap_server,
                on_start=[map_conversion],
            )
        )
    )

    return ld
