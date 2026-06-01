from pathlib import Path

from setuptools import find_packages, setup

package_name = "scene_graph_core"
root = Path(__file__).resolve().parent
readme_path = root / "README.md"

install_requires = [
    "networkx>=2.4",
    "numpy>=1.21,<2",
    "scipy>=1.7",
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "isort>=5.10",
        ],
        "visualization": ["matplotlib>=3.5"],
    },
    python_requires=">=3.8",
    zip_safe=True,
    maintainer="Carlos Cueto Zumaya",
    maintainer_email="carlos.cueto@example.com",
    description=(
        "Core library for 3D Scene Graph construction and manipulation in robotics."
    ),
    long_description=readme_path.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    tests_require=["pytest"],
    url="https://github.com/crcz25/3dsg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
