import os
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter, SetRemap
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions import LogInfo, GroupAction
from launch.substitutions import (
    TextSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    NotSubstitution,
    AndSubstitution,
    OrSubstitution,
    EqualsSubstitution,
    PythonExpression,
)
from launch.substitution import Substitution
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

############# Control robot with keyboard ###############
### $ ros2 run racing_bot_controller teleop_key_node ###
### $ ros2 run racing_bot_controller controller_node ###
#########################################################

### If defined, the maps are obtained from `ROS_MAP_FILES_DIR`
### Logs are stored in  `ROS_LOG_FILES_DIR`


def are_equal_substition(left, right) -> Substitution:
    """Use instead of launch.substitutions.EqualsSubstition, since ROS2 Humble doesn't have the aforementioned class.
    example:
        are_equal_substition(slam_mode, "localization")
    INSTEAD OF:
        EqualsSubstitution(slam_mode, "localization")
    """
    return PythonExpression(["'", left, "' == '", right, "'"])


def generate_launch_description():
    use_ekf_launch_arg = DeclareLaunchArgument(
        "use_ekf",
        default_value=TextSubstitution(text="true"),
        description="Use the extended kalman filter for the odometry data."
        "This will also activate and use the imu if available.",
    )
    slam_mode_launch_arg = DeclareLaunchArgument(
        "slam_mode",
        default_value=TextSubstitution(text="elsewhere"),
        choices=["mapping", "localization", "elsewhere", "disabled"],
        description="Which mode of the slam_toolbox to use: SLAM (=mapping), only localization (=localization),"
        + " on another device (=elsewhere) or don't use so map frame is odom frame (=disabled).",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )
    follow_traject_launch_arg = DeclareLaunchArgument(
        "follow_traject",
        default_value=TextSubstitution(text="false"),
        description="if true follows traject",
    )
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="Don't run the sensor nodes when its run on a rosbag",
    )

    use_ekf = LaunchConfiguration("use_ekf")
    slam_mode = LaunchConfiguration("slam_mode")
    map_file = LaunchConfiguration("map_file")
    follow_traject = LaunchConfiguration("follow_traject")
    play_rosbag = LaunchConfiguration("play_rosbag")

    log_messages = []
    log_messages.append(LogInfo(msg="\n=========================== Launch file logging ==========================="))

    try:
        map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        log_messages.append(LogInfo(msg="The map is loaded from the path in `ROS_MAP_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_MAP_FILES_DIR` is not set!"))
        map_files_dir = ""

    log_messages.append(
        LogInfo(
            msg="The extended Kalman filter will be used to estimate the position in the \odom frame...",
            condition=IfCondition(use_ekf),
        )
    )
    log_messages.append(LogInfo(msg="\n================================== END ==================================\n"))

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
        condition=UnlessCondition(play_rosbag),
    )

    # Sensor related
    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        condition=UnlessCondition(play_rosbag),
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        condition=UnlessCondition(play_rosbag),
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_imu"), "config", "imu_node.yaml"]),
        ],
        condition=UnlessCondition(play_rosbag),
    )
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("sllidar_ros2"), "launch", "sllidar_a1_launch.py"])
        ),
        condition=UnlessCondition(play_rosbag),
    )

    # State estimation related
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_odometry"), "config", "odometry_node.yaml"]),
            {"do_broadcast_transform": NotSubstitution(use_ekf)},  # Use either this or ekf transform (set in ekf.yaml)
        ],
    )
    velocity_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="velocity_ekf_node",
        parameters=[PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "ekf.yaml"])],
        # parameters=[PathJoinSubstitution(["/home/christiaan/thesis/robot_ws/src/racing_bot_bringup/config/ekf.yaml"])],
        condition=IfCondition(use_ekf),  # use_ekf
        remappings=[("odometry/filtered", "odometry/velocity_ekf")],
    )
    position_ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="position_ekf_node",
        parameters=[PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "ekf.yaml"])],
        # parameters=[PathJoinSubstitution(["/home/christiaan/thesis/robot_ws/src/racing_bot_bringup/config/ekf.yaml"])],
        condition=IfCondition(use_ekf),  # use_ekf
        remappings=[("odometry/filtered", "odometry/position_ekf")],
    )
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "slam_launch.py"]
            )
        ),
        launch_arguments={
            "localization": EqualsSubstitution(slam_mode, "localization"),
            "mapping": EqualsSubstitution(slam_mode, "mapping"),
            "map_file": map_file,
            "publish_tf": NotSubstitution(use_ekf),
        }.items(),
    )

    # Other
    trajectory_node = Node(
        package="racing_bot_trajectory_follower",
        executable="trajectory_follower_node",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_trajectory_follower"), "config", "trajectory_follower_node.yaml"]
            ),
        ],
        condition=IfCondition(AndSubstitution(follow_traject, NotSubstitution(play_rosbag))),
    )
    static_trans_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=("--frame-id map --child-frame-id odom").split(" "),
        condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
    )
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
            # arguments
            use_ekf_launch_arg,
            slam_mode_launch_arg,
            map_file_launch_arg,
            follow_traject_launch_arg,
            # log messages
            *log_messages,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            odometry_node,
            trajectory_node,
            imu_node,
            # ekf_node,
            velocity_ekf_node,
            position_ekf_node,
            # launch files
            lidar_launch,
            slam_launch,
            # static transforms
            static_trans_map_to_odom,
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
        ]
    )