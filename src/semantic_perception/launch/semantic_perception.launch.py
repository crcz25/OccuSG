from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("semantic_perception"), "config", "semantic_perception.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="semantic_perception",
                executable="semantic_perception_node",
                name="semantic_perception",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
