#!/usr/bin/env python3
"""
Example launch file for depth_to_pointcloud node.

This launch file demonstrates how to configure and launch the depth to
point cloud conversion node with custom parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for depth_to_pointcloud node."""
    # Declare launch arguments for dynamic configuration
    depth_image_topic_arg = DeclareLaunchArgument(
        "depth_image_topic",
        default_value="/camera/depth/image_raw",
        description="Topic name for depth image input",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/depth/camera_info",
        description="Topic name for camera info input",
    )

    output_frame_arg = DeclareLaunchArgument(
        "output_frame",
        default_value="camera_depth_optical_frame",
        description="Frame ID for the output point cloud",
    )

    pointcloud_topic_arg = DeclareLaunchArgument(
        "pointcloud_topic",
        default_value="/pointcloud",
        description="Topic name for point cloud output",
    )

    stride_arg = DeclareLaunchArgument(
        "stride",
        default_value="1",
        description=(
            "Downsampling stride (use every Nth pixel). "
            "Higher = faster but fewer points"
        ),
    )

    min_depth_arg = DeclareLaunchArgument(
        "min_depth",
        default_value="0.1",
        description="Minimum valid depth in meters",
    )

    max_depth_arg = DeclareLaunchArgument(
        "max_depth",
        default_value="10.0",
        description="Maximum valid depth in meters",
    )

    target_hz_arg = DeclareLaunchArgument(
        "target_hz",
        default_value="0.0",
        description="Point cloud output rate limit in Hz (0 = unlimited)",
    )

    sync_queue_size_arg = DeclareLaunchArgument(
        "sync_queue_size",
        default_value="10",
        description="Approximate sync queue size",
    )

    sync_slop_sec_arg = DeclareLaunchArgument(
        "sync_slop_sec",
        default_value="0.05",
        description="Approximate sync tolerance in seconds",
    )

    qos_depth_arg = DeclareLaunchArgument(
        "qos_depth",
        default_value="10",
        description="QoS queue depth for subscriptions and publisher",
    )

    input_qos_reliability_arg = DeclareLaunchArgument(
        "input_qos_reliability",
        default_value="best_effort",
        description="Input QoS reliability: best_effort or reliable",
    )

    output_qos_reliability_arg = DeclareLaunchArgument(
        "output_qos_reliability",
        default_value="reliable",
        description="Output QoS reliability: best_effort or reliable",
    )

    # Create the node with parameters
    depth_to_pointcloud_node = Node(
        package="point_cloud_generator",
        executable="depth_to_pointcloud",
        name="depth_to_pointcloud_node",
        output="screen",
        parameters=[
            {
                "depth_image_topic": LaunchConfiguration("depth_image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "output_frame": LaunchConfiguration("output_frame"),
                "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                "stride": LaunchConfiguration("stride"),
                "min_depth": LaunchConfiguration("min_depth"),
                "max_depth": LaunchConfiguration("max_depth"),
                "target_hz": LaunchConfiguration("target_hz"),
                "sync_queue_size": LaunchConfiguration("sync_queue_size"),
                "sync_slop_sec": LaunchConfiguration("sync_slop_sec"),
                "qos_depth": LaunchConfiguration("qos_depth"),
                "input_qos_reliability": LaunchConfiguration(
                    "input_qos_reliability"
                ),
                "output_qos_reliability": LaunchConfiguration(
                    "output_qos_reliability"
                ),
            }
        ],
        # Optional: remap topics if needed
        # remappings=[
        #     ('/camera/depth/image_raw', '/my_camera/depth/image'),
        # ]
    )

    return LaunchDescription(
        [
            depth_image_topic_arg,
            camera_info_topic_arg,
            output_frame_arg,
            pointcloud_topic_arg,
            stride_arg,
            min_depth_arg,
            max_depth_arg,
            target_hz_arg,
            sync_queue_size_arg,
            sync_slop_sec_arg,
            qos_depth_arg,
            input_qos_reliability_arg,
            output_qos_reliability_arg,
            depth_to_pointcloud_node,
        ]
    )
