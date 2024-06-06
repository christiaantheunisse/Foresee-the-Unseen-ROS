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
    do_visualize_launch_arg = DeclareLaunchArgument(
        "do_visualize",
        default_value=TextSubstitution(text="true"),
        description="If true, the planner node ",
    )
    do_triangulate_launch_arg = DeclareLaunchArgument(
        "do_triangulate",
        default_value=TextSubstitution(text="false"),
        description="If true, the planner node ",
    )

    do_visualize = LaunchConfiguration("do_visualize")
    do_triangulate = LaunchConfiguration("do_triangulate")

    log_messages = []
    try:
        log_files_dir = os.environ["ROS_LOG_FILES_DIR"]
        log_messages.append(LogInfo(msg="The log data is stored in the path in `ROS_LOG_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_LOG_FILES_DIR` is not set!"))
        log_files_dir = ""

    fov_node = Node(
        package="foresee_the_unseen",
        executable="fov_node",
        name="fov_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "fov_node.yaml"]),
            {
                "do_visualize": do_visualize,
                "error_models_directory": PathJoinSubstitution(
                    [FindPackageShare("foresee_the_unseen"), "resource", "error_models"]
                ),
            },
        ],
    )
    planner_node = Node(
        package="foresee_the_unseen",
        executable="planner_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "planner_node.yaml"]),
            {
                "use_triangulation": do_triangulate,
                "do_visualize": do_visualize,
                # "road_xml": PathJoinSubstitution(
                #     [FindPackageShare("foresee_the_unseen"), "resource", "road_structure_15_reduced_points.xml"]
                # ),
                "error_models_directory": PathJoinSubstitution(
                    [FindPackageShare("foresee_the_unseen"), "resource", "error_models"]
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
        # $ ros2 run tf2_ros static_transform_publisher --x 3.8 --y 0.13 --yaw -1.57 --frame-id map --child-frame-id planner
        static_trans_map_to_planner_frame = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=(
                f"--x {init_x} --y {init_y} --z 0 --roll 0 --pitch 0 --yaw {init_th} "
                + "--frame-id map --child-frame-id planner"
            ).split(" "),
        )

        return [static_trans_map_to_planner_frame]

    return LaunchDescription(
        [
            do_visualize_launch_arg,
            do_triangulate_launch_arg,
            *log_messages,
            fov_node,
            planner_node,
            OpaqueFunction(
                function=get_map_to_planner_frame_tf,
            ),
        ]
    )
