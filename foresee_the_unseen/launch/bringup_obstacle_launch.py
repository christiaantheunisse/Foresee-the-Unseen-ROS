import os
from launch import LaunchDescription, LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions.opaque_function import OpaqueFunction
from launch.actions import LogInfo, GroupAction
from launch.substitutions import (
    TextSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    NotSubstitution,
    AndSubstitution,
    OrSubstitution,
    EqualsSubstitution,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    namespace_launch_arg = DeclareLaunchArgument(
        "namespace",
        default_value=TextSubstitution(text="obstacle"),
        description="The namespace used for the maps and all the topics.",
    )
    use_ekf_launch_arg = DeclareLaunchArgument(
        "use_ekf",
        default_value=TextSubstitution(text="true"),
        description="Use the extended kalman filter for the odometry data."
        "This will also activate and use the imu if available.",
    )
    lidar_reverse_launch_arg = DeclareLaunchArgument(
        "lidar_reverse",
        default_value=TextSubstitution(text="false"),
        description="True if the lidar is mounted with the motor facing forward",
    )

    namespace = LaunchConfiguration("namespace")
    use_ekf = LaunchConfiguration("use_ekf")
    lidar_reverse = LaunchConfiguration("lidar_reverse")

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
    )

    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        parameters=[
            {
                "left_reverse": True,
                "right_reverse": True,
            }
        ],
        namespace=namespace,
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_encoder"), "config", "encoder_node.yaml"]),
            {
                "left_pin_a": 21,
                "left_pin_b": 19,
                "right_pin_a": 13,
                "right_pin_b": 12,
            },
        ],
        namespace=namespace,
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
        namespace=namespace,
    )

    def namespace_as_string(context: LaunchContext, namespace: LaunchConfiguration):
        """In this function the namespace is available as string and can be used for the frames"""
        namespace_str = namespace.perform(context)

        odometry_node = Node(
            package="racing_bot_odometry",
            executable="odometry_node",
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_odometry"), "config", "odometry_node.yaml"]),
                {
                    "do_broadcast_transform": True,
                    "odom_frame": f"{namespace_str}/odom",
                    "base_frame": f"{namespace_str}/base_link",
                    "wheel_radius": 0.030,
                    "wheel_base": 0.103,
                },
            ],
            namespace=namespace,
        )
        trajectory_node = Node(
            package="racing_bot_trajectory_follower",
            executable="trajectory_follower_node",
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("racing_bot_trajectory_follower"), "config", "trajectory_follower_node.yaml"]
                ),
                {
                    "follow_mode": "position",
                    "odom_frame": f"{namespace_str}/odom",
                    "wheel_base_width": 0.103,
                    "steering_k": 0.5,
                },
            ],
            namespace=namespace,
        )
        ekf_node = Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_node",
            parameters=[
                PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "ekf_obstacle.yaml"]),
                {
                    "odom_frame": f"{namespace_str}/odom",
                    "base_link_frame": f"{namespace_str}/base_link",
                },
            ],
            condition=IfCondition(use_ekf),  # use_ekf
            namespace=namespace,
        )
        lidar_launch = GroupAction(
            actions=[
                PushRosNamespace(namespace),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([FindPackageShare("sllidar_ros2"), "launch", "sllidar_a1_launch.py"])
                    ),
                    launch_arguments=[("frame_id", f"{namespace_str}/laser")],
                ),
            ]
        )

        # Needed: transform from map to odom: SLAM or static, but should be handle on the laptop because it is based on
        #  the trajectory for the obstacle

        static_trans_base_link_to_laser = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=(
                f"--x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 3.14 "
                + f"--frame-id {namespace_str}/base_link --child-frame-id {namespace_str}/laser"
            ).split(" "),
            condition=UnlessCondition(lidar_reverse),
        )
        static_trans_base_link_to_laser_reverse = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=(
                f"--x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 "
                + f"--frame-id {namespace_str}/base_link --child-frame-id {namespace_str}/laser"
            ).split(" "),
            condition=IfCondition(lidar_reverse),
        )

        return [
            odometry_node,
            trajectory_node,
            ekf_node,
            lidar_launch,
            static_trans_base_link_to_laser,
            static_trans_base_link_to_laser_reverse,
        ]

    return LaunchDescription(
        [
            # arguments
            namespace_launch_arg,
            use_ekf_launch_arg,
            lidar_reverse_launch_arg,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            controller_node,
            OpaqueFunction(function=namespace_as_string, args=[namespace]),
        ]
    )
