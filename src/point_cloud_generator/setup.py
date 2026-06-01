import os
from glob import glob

from setuptools import find_packages, setup

package_name = "point_cloud_generator"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="devuser",
    maintainer_email="crcueto25@gmail.com",
    description="ROS 2 package for converting depth images to PointCloud2 messages",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "depth_to_pointcloud = point_cloud_generator.depth_to_pointcloud_node:main",
        ],
    },
)
