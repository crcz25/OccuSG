import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("semantic_perception")
    param_file = os.path.join(pkg_share, "config", "config.yaml")
    rviz_config = os.path.join(pkg_share, "config", "rviz.rviz")

    with open(param_file, "r") as f:
        all_params = yaml.safe_load(f)
        p_sem = all_params["semantic_node"]["ros__parameters"]

    # Semantic perception
    perception = Node(
        package="semantic_perception",
        executable="semantic_perception",
        name="semantic_node",
        output="screen",
        parameters=[p_sem],
    )

    # RViz
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([perception, rviz])
