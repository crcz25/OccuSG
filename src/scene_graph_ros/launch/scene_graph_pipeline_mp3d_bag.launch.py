"""Launch the scene-graph pipeline against a recorded MP3D ROS bag."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def _validate_launch_arguments(context, *args, **kwargs):
    bag_path = LaunchConfiguration("bag_path").perform(context)
    scan_id = LaunchConfiguration("scan_id").perform(context)
    if not bag_path:
        raise RuntimeError("bag_path must point to the directory containing MP3D bags")
    if not scan_id:
        raise RuntimeError("scan_id must name the MP3D bag to play")

    bag_uri = os.path.join(bag_path, scan_id)
    if not os.path.exists(bag_uri):
        raise RuntimeError(f"MP3D bag path does not exist: {bag_uri}")
    return []


def generate_launch_description() -> LaunchDescription:
    scene_graph_share = get_package_share_directory("scene_graph_ros")
    inc_dude_share = get_package_share_directory("incremental_dude_ros2")

    default_params = os.path.join(
        scene_graph_share, "config", "scene_graph_pipeline_params_mp3d.yaml"
    )
    default_rviz_config = os.path.join(
        scene_graph_share, "config", "rviz_mp3d_bag.rviz"
    )
    default_inc_dude_params = os.path.join(
        inc_dude_share, "config", "inc_dude_params.yaml"
    )

    bag_path = LaunchConfiguration("bag_path")
    scan_id = LaunchConfiguration("scan_id")
    params = LaunchConfiguration("params_file")
    inc_dude_params = LaunchConfiguration("inc_dude_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    region_map_topic = LaunchConfiguration("region_map_topic")
    stable_regions_topic = LaunchConfiguration("stable_regions_topic")
    enable_profiling = LaunchConfiguration("enable_profiling")
    profiling_output_path = LaunchConfiguration("profiling_output_path")
    profiling_run_name = LaunchConfiguration("profiling_run_name")
    profiling_save_on_shutdown = LaunchConfiguration("profiling_save_on_shutdown")
    profiling_discard_first_n = LaunchConfiguration("profiling_discard_first_n")

    bag_uri = PathJoinSubstitution([bag_path, scan_id])
    export_json_path = PathJoinSubstitution([bag_path, scan_id, "scene_graph.json"])
    default_profiling_output_path = PathJoinSubstitution(
        [bag_path, scan_id, "profiling"]
    )

    rviz_enabled = PythonExpression(["'", use_rviz, "' == 'true'"])

    launch_args = [
        DeclareLaunchArgument(
            "bag_path",
            description="Directory containing the MP3D bag folders.",
        ),
        DeclareLaunchArgument(
            "scan_id",
            description="MP3D bag folder/name to play from bag_path.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Unified ROS parameter YAML for the MP3D pipeline.",
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
            description="Use rosbag /clock published by ros2 bag play --clock.",
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
        DeclareLaunchArgument(
            "enable_profiling",
            default_value="false",
            description="Enable lightweight runtime profiling JSON output.",
        ),
        DeclareLaunchArgument(
            "profiling_output_path",
            default_value=default_profiling_output_path,
            description="Directory for per-process profiling JSON files.",
        ),
        DeclareLaunchArgument(
            "profiling_run_name",
            default_value=[scan_id, "_region"],
            description="Run name prefix for profiling files.",
        ),
        DeclareLaunchArgument(
            "profiling_save_on_shutdown",
            default_value="true",
            description="Save profiling JSON when each process shuts down.",
        ),
        DeclareLaunchArgument(
            "profiling_discard_first_n",
            default_value="5",
            description="Warmup samples to discard from profiling summaries.",
        ),
    ]

    profiling_params = {
        "enable_profiling": enable_profiling,
        "profiling_output_path": profiling_output_path,
        "profiling_run_name": profiling_run_name,
        "profiling_save_on_shutdown": profiling_save_on_shutdown,
        "profiling_discard_first_n": profiling_discard_first_n,
    }

    bag_player = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_uri, "--clock"],
        name="mp3d_bag_player",
        output="screen",
    )

    point_cloud_generator = Node(
        package="point_cloud_generator",
        executable="depth_to_pointcloud",
        name="point_cloud_generator_node",
        output="log",
        parameters=[
            params,
            {
                "use_sim_time": use_sim_time,
                "depth_image_topic": "/depth",
                "camera_info_topic": "/depth/camera_info",
                "output_frame": "depth_camera_optical_frame",
                "pointcloud_topic": "/pointcloud",
                "input_qos_reliability": "reliable",
                "output_qos_reliability": "reliable",
                **profiling_params,
            },
        ],
    )

    octomap_server = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server",
        output="log",
        parameters=[
            params,
            {"use_sim_time": use_sim_time, "frame_id": "odom", **profiling_params},
        ],
        remappings=[("cloud_in", "/pointcloud")],
    )

    map_conversion = Node(
        package="mapconversion",
        executable="map_conversion_oct_node",
        name="map_conversion_node",
        output="log",
        parameters=[
            params,
            {"use_sim_time": use_sim_time, "map_frame": "odom", **profiling_params},
        ],
        remappings=[("octomap", "octomap_full")],
    )

    perception = Node(
        package="semantic_perception",
        executable="semantic_perception",
        name="semantic_node",
        output="log",
        parameters=[
            params,
            {
                "use_sim_time": use_sim_time,
                "rgb_topic": "/rgb",
                "depth_topic": "/depth",
                "depth_info_topic": "/depth/camera_info",
                "target_frame": "odom",
            },
        ],
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
                **profiling_params,
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
                "odom_topic": "/odom",
                "stable_regions_topic": stable_regions_topic,
                "fixed_frame_id": "odom",
                "export_json_path": export_json_path,
                "export_json_on_shutdown": True,
                **profiling_params,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="log",
        condition=IfCondition(rviz_enabled),
    )

    launch_actions = list(launch_args)
    launch_actions.append(OpaqueFunction(function=_validate_launch_arguments))
    launch_actions.extend(
        [
            point_cloud_generator,
            octomap_server,
            map_conversion,
            perception,
            inc_dude,
            scene_graph_region,
            rviz,
            TimerAction(period=5.0, actions=[bag_player]),
        ]
    )

    return LaunchDescription(launch_actions)
