"""Canonical launch file for the minimal scene-graph pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _start_after(target_action: Node, actions: list) -> RegisterEventHandler:
    """Create an event handler that starts actions when target_action starts."""
    return RegisterEventHandler(
        OnProcessStart(target_action=target_action, on_start=actions)
    )


def generate_launch_description() -> LaunchDescription:
    scene_graph_share = get_package_share_directory("scene_graph_ros")
    inc_dude_share = get_package_share_directory("incremental_dude_ros2")

    default_params = os.path.join(
        scene_graph_share, "config", "scene_graph_pipeline_params.tbot3.yaml"
    )
    default_rviz_config = os.path.join(scene_graph_share, "config", "rviz_tbot.rviz")
    default_inc_dude_params = os.path.join(
        inc_dude_share, "config", "inc_dude_params.yaml"
    )

    params = LaunchConfiguration("params_file")
    inc_dude_params = LaunchConfiguration("inc_dude_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    region_map_topic = LaunchConfiguration("region_map_topic")
    stable_regions_topic = LaunchConfiguration("stable_regions_topic")

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Unified ROS parameter YAML for the minimal pipeline.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_config,
            description="RViz config file.",
        ),
        DeclareLaunchArgument(
            "inc_dude_params_file",
            default_value=default_inc_dude_params,
            description="ROS parameter YAML for incremental_dude_ros2.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to start RViz2.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulated ROS clock.",
        ),
        DeclareLaunchArgument(
            "region_map_topic",
            default_value="/mapUAV",
            description="OccupancyGrid topic used as input to incremental_dude_ros2.",
        ),
        DeclareLaunchArgument(
            "stable_regions_topic",
            default_value="/dude/regions",
            description="Region2DArray topic consumed by scene_graph_region.",
        ),
    ]

    point_cloud_generator = Node(
        package="point_cloud_generator",
        executable="depth_to_pointcloud",
        name="point_cloud_generator_node",
        output="log",
        parameters=[params, {"use_sim_time": use_sim_time}],
    )

    octomap_server = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server",
        output="log",
        parameters=[params, {"use_sim_time": use_sim_time}],
        remappings=[("cloud_in", "/pointcloud")],
    )

    map_conversion = Node(
        package="mapconversion",
        executable="map_conversion_oct_node",
        name="map_conversion_node",
        output="log",
        parameters=[params, {"use_sim_time": use_sim_time}],
        remappings=[("octomap", "octomap_full")],
    )

    perception = Node(
        package="semantic_perception",
        executable="semantic_perception",
        name="semantic_node",
        output="log",
        parameters=[params, {"use_sim_time": use_sim_time}],
    )

    inc_dude = Node(
        package="incremental_dude_ros2",
        executable="inc_dude",
        name="incremental_decomposer",
        output="screen",
        parameters=[
            inc_dude_params,
            {
                "occupancy_grid_topic": region_map_topic,
                "use_sim_time": use_sim_time,
            },
        ],
        remappings=[("/dude/regions", stable_regions_topic)],
    )

    scene_graph_region = Node(
        package="scene_graph_ros",
        executable="scene_graph_region",
        name="scene_graph_region",
        output="screen",
        parameters=[
            params,
            {
                "use_sim_time": use_sim_time,
                "stable_regions_topic": stable_regions_topic,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="log",
        condition=IfCondition(use_rviz),
    )

    launch_actions = list(launch_args)
    launch_actions.append(point_cloud_generator)
    launch_actions.append(_start_after(point_cloud_generator, [octomap_server]))
    launch_actions.append(_start_after(octomap_server, [map_conversion]))
    launch_actions.append(_start_after(map_conversion, [perception, inc_dude]))
    launch_actions.append(_start_after(inc_dude, [scene_graph_region]))
    launch_actions.append(_start_after(scene_graph_region, [rviz]))

    return LaunchDescription(launch_actions)
