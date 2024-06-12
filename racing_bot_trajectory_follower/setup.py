import os
from glob import glob
from setuptools import find_packages, setup


package_name = "racing_bot_trajectory_follower"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config/*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="christiaan",
    maintainer_email="christiaantheunisse@hotmail.nl",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "trajectory_follower_node = racing_bot_trajectory_follower.trajectory_follower_node:main",
            "trajectory_follower_node2 = racing_bot_trajectory_follower.trajectory_follower_node2:main",
            "trajectory_follower_node_obstacle = racing_bot_trajectory_follower.trajectory_follower_node_obstacle:main",
        ],
    },
)
