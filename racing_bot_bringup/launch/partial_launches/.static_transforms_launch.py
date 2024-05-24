from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    TextSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    AndSubstitution,
    NotSubstitution,
)
from launch.conditions import IfCondition
from launch.actions.opaque_function import OpaqueFunction

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Robot transforms
    static_trans_base_link_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=("--x 0.085 --yaw 3.14 --frame-id base_link --child-frame-id laser").split(" "),
    )
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=("--x 0.025 --y -0.038 --yaw -1.57080 --frame-id base_link --child-frame-id imu_link").split(" "),
    )

    return LaunchDescription(
        [   
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
        ]
    )
