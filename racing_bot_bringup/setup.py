from setuptools import find_packages, setup
import os
from glob import glob

package_name = "racing_bot_bringup"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config/*.yaml"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz*"))),
        (
            os.path.join("share", package_name, "launch", "slam_toolbox"),
            glob(os.path.join("launch", "slam_toolbox", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "launch", "partial_launches"),
            glob(os.path.join("launch", "partial_launches", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="C.Theunisse@student.tudelft.nl",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
