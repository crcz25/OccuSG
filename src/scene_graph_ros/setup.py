import os
from glob import glob

from setuptools import find_packages, setup

package_name = "scene_graph_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
    ],
    install_requires=[
        "setuptools",
        "scene_graph_core",
        "numpy>=1.21,<2",
    ],
    scripts=["scripts/aggregate_runtime_profiles.py"],
    zip_safe=True,
    maintainer="devuser",
    maintainer_email="crcueto25@gmail.com",
    description="ROS 2 scene graph integration and orchestration pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scene_graph_region = scene_graph_ros.scene_graph_region:main",
            # Individual nodes for testing/debugging
            "visualization_node = scene_graph_ros.visualization_node:main",
            "scene_graph_pose_diagnostics = scene_graph_ros.pose_diagnostics_node:main",
            # Utility nodes
            "robot_trajectory_node = scene_graph_ros.robot_trajectory:main",
        ],
    },
)
