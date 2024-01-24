import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.substitutions import TextSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # SetParameter(name='use_sim_time', value=True)
    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
    )

    # Start the necessary nodes
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
    )

    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        ros_arguments=["-p", "do_broadcast_transform:=false", # Use either this or ekf transform (set in ekf.yaml)
                    #    "-p", "pose_variances:=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                    #    "-p", "twist_variances:=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                       "-p", f"pose_variances:={[0.0001]*6}",
                       "-p", f"twist_variances:={[0.0001]*6}",
                       ],  
        output="screen",
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
    )
    # Extended Kalman Filter: https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
    # pkg_foresee_the_unseen = FindPackageShare("foresee_the_unseen")
    # path_ekf_yaml = PathJoinSubstitution([pkg_foresee_the_unseen, "config", "ekf.yaml"])
    path_ekf_yaml = "/home/ubuntu/thesis_ws/src/foresee_the_unseen/config/ekf.yaml" # don't need to build
    robot_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="my_ekf_filter_node",
        output="screen",
        parameters=[path_ekf_yaml],
    )
    visualization_node = Node(
        package="foresee_the_unseen",
        executable="visualization_node",
    )

    # Static transforms
    # ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /odom --child-frame-id /map
    static_trans_odom_to_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "odom",
            "--child-frame-id", "map",
        ],
    )
    # ros2 run tf2_ros static_transform_publisher --x -0.1 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /laser --child-frame-id /base_link
    static_trans_base_link_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.85",
            "--y", "0",
            "--z", "0",
            "--roll", "3.14",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "laser",
        ],
    )
    # ros2 run tf2_ros static_transform_publisher --x -0.035 --y 0.04 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /base_link --child-frame-id /imu_link
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.035",
            "--y", "0.04",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "imu_link",
        ],
    )

    return LaunchDescription(
        [
            # arguments
            # commands
            start_pigpiod,
            # nodes
            encoder_node,
            odometry_node,
            imu_node,
            robot_localization_node,
            visualization_node,
            # launch files
            # transforms
            static_trans_odom_to_map,
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
        ]
    )
