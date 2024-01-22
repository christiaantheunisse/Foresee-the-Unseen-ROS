import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.substitutions import TextSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Values should be `true` or `1` and `false` or `0`
    use_lidar_launch_arg = DeclareLaunchArgument("use_lidar", default_value=TextSubstitution(text="true"))

    # To start the `pigpiod package`, necessary for communication with through the motor hat
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
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
    )

    # Launch lidar nodes if `use_lidar` == True
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("sllidar_ros2"), "launch/sllidar_a1_launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("use_lidar")),
    )

    # Static transforms
    # ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /odom --child-frame-id /map
    static_trans_odom_to_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "odom", "map"],
    )

    # ros2 run tf2_ros static_transform_publisher --x -0.1 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /laser --child-frame-id /base_link
    static_trans_laser_to_base_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.1", "0", "0", "3.14", "0", "0", "base_link", "laser"],
    )

    return LaunchDescription(
        [
            # arguments
            use_lidar_launch_arg,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            odometry_node,
            controller_node,
            # launch files
            lidar_launch,
            # transforms
            static_trans_odom_to_map,
            static_trans_laser_to_base_link,
        ]
    )
