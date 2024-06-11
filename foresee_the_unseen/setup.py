import os
from glob import glob
from setuptools import find_packages, setup

package_name = "foresee_the_unseen"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        (
            os.path.join("share", package_name, "launch", "slam_toolbox"),
            glob(os.path.join("launch", "slam_toolbox", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "launch", "partial_launches"),
            glob(os.path.join("launch", "partial_launches", "*launch.[pxy][yma]*")),
        ),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config/*.yaml"))),
        (os.path.join("share", package_name, "resource"), glob(os.path.join("resource", "*.yaml"))),
        (os.path.join("share", package_name, "resource"), glob(os.path.join("resource", "*.xml"))),
        (
            os.path.join("share", package_name, "resource", "error_models"),
            glob(os.path.join("resource", "error_models", "*.pickle")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_node = foresee_the_unseen.planner_node:main",
            "fov_node = foresee_the_unseen.fov_node:main",
            "topics_to_disk_node = foresee_the_unseen.topics_to_disk_node:main",
            "logging_node = foresee_the_unseen.logging_node:main",
            "obstacle_trajectories_node = foresee_the_unseen.obstacle_trajectories_node:main",
            "test_straight_trajectories_node = foresee_the_unseen.test_straight_trajectories_node:main",
        ],
    },
)
