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
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="If false, bring up the real robot.",
    )
    play_rosbag = LaunchConfiguration("play_rosbag")

    log_messages = []
    try:
        log_files_dir = os.environ["ROS_LOG_FILES_DIR"]
        log_messages.append(LogInfo(msg="The log data is stored in the path in `ROS_LOG_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_LOG_FILES_DIR` is not set!"))
        log_files_dir = ""

    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    planner_node = Node(
        package="foresee_the_unseen",
        executable="planner_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "planner_node.yaml"]),
            {
                "road_xml": PathJoinSubstitution(
                    [FindPackageShare("foresee_the_unseen"), "resource", "road_structure_15_reduced_points.xml"]
                ),
                "log_directory": PathJoinSubstitution(log_files_dir),
            },
        ],
    )

    # $ ros2 run tf2_ros static_transform_publisher --x 3 --y 2 --z 0 --yaw 1.57 --pitch 0 --roll 0 --frame-id map --child-frame-id planner
    static_trans_map_to_planner_frame = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "3.80",  # 1.80
            "--y",
            "0.13",  # 0.13
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "-1.57079632679",
            "--frame-id",
            "map",
            "--child-frame-id",
            "planner",
        ],
    )

    return LaunchDescription(
        [
            play_rosbag_launch_arg,
            *log_messages,
            do_use_sim_time,
            planner_node,
            static_trans_map_to_planner_frame,
        ]
    )
