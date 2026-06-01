from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("incremental_dude_ros2"),
        "config",
        "inc_dude_params.yaml",
    )

    decomp_threshold_arg = DeclareLaunchArgument(
        "decomp_threshold",
        default_value="3.0",
        description="Decomposition threshold in meters",
    )

    occupancy_grid_topic_arg = DeclareLaunchArgument(
        "occupancy_grid_topic",
        default_value="/mapUAV",
        description="OccupancyGrid topic to subscribe to",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation time",
    )

    inc_dude_node = Node(
        package="incremental_dude_ros2",
        executable="inc_dude",
        name="incremental_decomposer",
        output="screen",
        parameters=[
            params_file,
            {
                "decomp_threshold": LaunchConfiguration("decomp_threshold"),
                "occupancy_grid_topic": LaunchConfiguration("occupancy_grid_topic"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    return LaunchDescription(
        [
            decomp_threshold_arg,
            occupancy_grid_topic_arg,
            use_sim_time_arg,
            inc_dude_node,
        ]
    )
