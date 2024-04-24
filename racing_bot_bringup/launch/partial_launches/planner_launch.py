import os
import yaml
from launch import LaunchDescription, LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions.opaque_function import OpaqueFunction
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
    # play_rosbag_launch_arg = DeclareLaunchArgument(
    #     "play_rosbag",
    #     default_value=TextSubstitution(text="false"),
    #     description="If false, bring up the real robot.",
    # )
    # play_rosbag = LaunchConfiguration("play_rosbag")

    log_messages = []
    try:
        log_files_dir = os.environ["ROS_LOG_FILES_DIR"]
        log_messages.append(LogInfo(msg="The log data is stored in the path in `ROS_LOG_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_LOG_FILES_DIR` is not set!"))
        log_files_dir = ""

    # do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

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


    def get_map_to_planner_frame_tf(context: LaunchContext):
        commonroad_config_yamlpath = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", "commonroad_scenario.yaml"]
        ).perform(context)
        with open(commonroad_config_yamlpath) as f:
            obstacle_config = yaml.safe_load(f)
        pose = obstacle_config["pose_of_planner_in_map_frame"]
        assert len(pose) == 3, f"Pose should be [x, y, theta]; {pose=}"

        init_x, init_y, init_th = pose
        static_trans_map_to_planner_frame = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=(
                f"--x {init_x} --y {init_y} --z 0 --roll 0 --pitch 0 --yaw {init_th} " +
                "--frame-id map --child-frame-id planner"
            ).split(" "),
        )

        return [static_trans_map_to_planner_frame]

    return LaunchDescription(
        [
            # play_rosbag_launch_arg,
            *log_messages,
            # do_use_sim_time,
            planner_node,
            OpaqueFunction(
                function=get_map_to_planner_frame_tf,
            ),
        ]
    )
