from glob import glob
from setuptools import find_packages, setup

package_name = "semantic_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/prompts", glob("prompts/*.csv")),
    ],
    install_requires=["setuptools", "numpy", "Pillow"],
    zip_safe=True,
    maintainer="Semantic Perception Maintainer",
    maintainer_email="maintainer@example.com",
    description="Synchronized RGB-D proposals with CLIP visual embeddings.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "semantic_perception_node = semantic_perception.node:main",
            "semantic_perception_standalone = semantic_perception.standalone:main",
        ],
    },
)
