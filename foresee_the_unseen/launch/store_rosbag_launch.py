import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.substitutions import TextSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
    )

    # Start the necessary nodes
    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
    )

    # $ ros2 run racing_bot_controller teleop_key_node
    # teleop_key_node = Node(
    #     package="racing_bot_controller",
    #     executable="teleop_key_node",
    #     condition=IfCondition(LaunchConfiguration("use_keys")),
    #     output='screen',
    # )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("sllidar_ros2"), "launch/sllidar_a1_launch.py")
        ),
    )

    return LaunchDescription(
        [
            # arguments
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            controller_node,
            imu_node,
            # launch files
            lidar_launch,
        ]
    )
