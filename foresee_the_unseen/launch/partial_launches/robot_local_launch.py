import os
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions import LogInfo
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
    # Values should be `true` or `1` and `false` or `0`
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="If false, bring up the real robot.",
    )
    use_imu_launch_arg = DeclareLaunchArgument(
        "use_imu",
        default_value=TextSubstitution(text="true"),
        description="Whether to use the IMU for the local localization. This will also start the extended kalman"
        + "filter node and the odometry node will not publish a transform",
    )
    slam_mode_launch_arg = DeclareLaunchArgument(
        "slam_mode",
        default_value=TextSubstitution(text="disabled"),
        choices=["mapping", "localization", "disabled"],
        description="Which mode of the slam_toolbox to use: SLAM, only localization or don't use.",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )

    log_messages = []
    try:
        map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        log_messages.append(
            LogInfo(msg="[localization_launch.py] The map is loaded from the path in `ROS_MAP_FILES_DIR`")
        )
    except KeyError:
        log_messages.append(LogInfo(msg="[localization_launch.py] Environment variable ROS_MAP_FILES_DIR is not set!"))
        map_files_dir = ""

    play_rosbag = LaunchConfiguration("play_rosbag")
    use_imu = LaunchConfiguration("use_imu")
    slam_mode = LaunchConfiguration("slam_mode")
    map_file = LaunchConfiguration("map_file")

    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    # ONLY used on real robot
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_imu"), "config", "imu_node.yaml"]),
        ],
        condition=IfCondition(AndSubstitution(NotSubstitution(play_rosbag), use_imu)),  # not play_rosbag and use_imu
    )
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("sllidar_ros2"), "launch", "sllidar_a1_launch.py"])
        ),
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )

    # Used ANYWAY
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_odometry"), "config", "odometry_node.yaml"]),
            {"do_broadcast_transform": NotSubstitution(use_imu)},  # Use either this or ekf transform (set in ekf.yaml)
        ],
    )
    local_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="my_ekf_filter_node",
        output="screen",
        parameters=[PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "ekf.yaml"])],
        condition=IfCondition(use_imu),
    )
    global_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # This launch file from the slam_toolbox is downloaded from github, because the version shipped with apt
            # does not provide the same arguments.
            PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "localization_launch.py"]
            )
        ),
        launch_arguments={
            "slam_params_file": PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_localization.yaml"]
            ),
            "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
        }.items(),
        condition=IfCondition(EqualsSubstitution(slam_mode, "localization")),  # slam_mode == "localization"
    )

    global_localization_and_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # This launch file from the slam_toolbox is downloaded from github, because the version shipped with apt
            # does not provide the same arguments.
            PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "online_async_launch.py"]
            )
        ),
        launch_arguments={
            "slam_params_file": PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_online_async.yaml"]
            ),
        }.items(),
        condition=IfCondition(EqualsSubstitution(slam_mode, "mapping")),  # slam_mode = "mapping"
    )

    # Static transforms
    # $ ros2 run tf2_ros static_transform_publisher --x 0.0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id map --child-frame-id odom
    static_trans_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "odom",
        ],
        condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
    )
    # $ ros2 run tf2_ros static_transform_publisher --x 0.085 --y 0 --z 0 --yaw 0 --pitch 0 --roll 3.14 --frame-id laser --child-frame-id base_link
    static_trans_base_link_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.085",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "3.14",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "laser",
        ],
    )
    # The angular velocity and linear acceleration are correctly transformed, but the orientation is wrong
    # $ ros2 run tf2_ros static_transform_publisher --x 0.025 --y -0.038 --z 0 --yaw 1.57080 --pitch 0 --roll 0 --frame-id base_link --child-frame-id imu_link
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.025",
            "--y",
            "-0.038",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "1.57080",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "imu_link",
        ],
    )
    return LaunchDescription(
        [
            # arguments
            play_rosbag_launch_arg,
            use_imu_launch_arg,
            slam_mode_launch_arg,
            map_file_launch_arg,
            # log messages
            *log_messages,
            # parameters
            do_use_sim_time,
            # commands
            # nodes
            encoder_node,
            odometry_node,
            imu_node,
            local_localization_node,
            # launch files
            lidar_launch,
            global_localization_launch,
            global_localization_and_mapping_launch,
            # transforms
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
            static_trans_map_to_odom,
        ]
    )
