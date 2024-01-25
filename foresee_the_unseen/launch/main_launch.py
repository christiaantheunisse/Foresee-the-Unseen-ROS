import os

from ament_index_python import get_package_share_directory

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
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

############# Control robot with keyboard ###############
### $ ros2 run racing_bot_controller teleop_key_node ###
#########################################################

################ Kill all ROS processes #####################
### $ kill $(ps aux | grep "ros-args" | awk '{print $2}') ###
#############################################################

################ Play Ros Bag ##################
### $ ros2 bag play -l just_on_desk/ --clock ###
################################################

################ Record Ros Bag ##################
### $ ---------------------------------------- ###
##################################################

######## Get tf2_ros tree ##############
### $ ros2 run tf2_tools view_frames ###
########################################


def generate_launch_description():
    # Values should be `true` or `1` and `false` or `0`
    play_rosbag_launch_arg = DeclareLaunchArgument("play_rosbag", default_value=TextSubstitution(text="false"))
    record_rosbag_launch_arg = DeclareLaunchArgument("record_rosbag", default_value=TextSubstitution(text="false"))
    use_ekf_launch_arg = DeclareLaunchArgument("use_ekf", default_value=TextSubstitution(text="false"))
    play_rosbag = LaunchConfiguration("play_rosbag")
    record_rosbag = LaunchConfiguration("record_rosbag")
    use_ekf = LaunchConfiguration("use_ekf")

    log_messages = []
    log_messages.append(LogInfo(msg="\n=========================== Launch file logging ==========================="))
    log_messages.append(
        LogInfo(msg="Only starting the nodes required to run on a rosbag...", condition=IfCondition(play_rosbag))
    )
    log_messages.append(
        LogInfo(msg="Starting all the nodes necessary to record a rosbag...", condition=IfCondition(record_rosbag))
    )
    log_messages.append(
        LogInfo(
            msg="The extended Kalman filter will be used to estimate the position in the \odom frame...",
            condition=IfCondition(use_ekf),
        )
    )
    log_messages.append(
        LogInfo(
            msg="ERROR! It is undefined to have both `play_rosbag:=True` and `record_rosbag:=True`",
            condition=IfCondition(AndSubstitution(play_rosbag, record_rosbag)),
        )
    )
    log_messages.append(LogInfo(msg="\n================================== END ==================================\n"))


    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
        # condition=IfCondition(OrSubstitution(NotSubstitution(play_rosbag), record_rosbag)),
    )

    # ONLY used on real robot
    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
        condition=IfCondition(
            AndSubstitution(NotSubstitution(play_rosbag), OrSubstitution(record_rosbag, use_ekf))
        ),  # not play_rosbag and (record_rosbag or use_ekf)
    )
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("sllidar_ros2"), "launch/sllidar_a1_launch.py")
        ),
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )

    # Used ANYWAY
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        parameters=[
            {"do_broadcast_transform": NotSubstitution(use_ekf)},  # Use either this or ekf transform (set in ekf.yaml)
            {"pose_variances": [0.0001] * 6},
            {"twist_variances": [0.0001] * 6},
        ],
        # output="screen",
        # emulate_tty=True,
    )
    visualization_node = Node(
        package="foresee_the_unseen",
        executable="visualization_node",
    )
    # Extended Kalman Filter: https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
    # pkg_foresee_the_unseen = FindPackageShare("foresee_the_unseen")
    # path_ekf_yaml = PathJoinSubstitution([pkg_foresee_the_unseen, "config", "ekf.yaml"])
    path_ekf_yaml = "/home/ubuntu/thesis_ws/src/foresee_the_unseen/config/ekf.yaml"  # don't need to build
    robot_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="my_ekf_filter_node",
        output="screen",
        parameters=[path_ekf_yaml],
        condition=IfCondition(use_ekf),  # use_ekf
    )

    # Static transforms
    # $ ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /map --child-frame-id /odom
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
    )
    # $ ros2 run tf2_ros static_transform_publisher --x 0.085 --y 0 --z 0 --yaw 0 --pitch 0 --roll 3.14 --frame-id /laser --child-frame-id /base_link
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
    # $ ros2 run tf2_ros static_transform_publisher --x -0.035 --y 0.04 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /base_link --child-frame-id /imu_link
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.035",
            "--y",
            "0.04",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
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
            record_rosbag_launch_arg,
            use_ekf_launch_arg,
            # log messages
            *log_messages,
            # parameters
            do_use_sim_time,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            odometry_node,
            controller_node,
            visualization_node,
            imu_node,
            robot_localization_node,
            # launch files
            lidar_launch,
            # transforms
            static_trans_map_to_odom,
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
        ]
    )
