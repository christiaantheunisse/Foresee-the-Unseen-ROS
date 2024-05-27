import os
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    TextSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    AndSubstitution,
    NotSubstitution,
    EqualsSubstitution,
)
from launch.conditions import IfCondition
from launch.actions.opaque_function import OpaqueFunction

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_mode_launch_arg = DeclareLaunchArgument(
        "slam_mode",
        default_value=TextSubstitution(text="elsewhere"),
        choices=["mapping", "localization", "elsewhere", "disabled"],
        description="Which mode of the slam_toolbox to use: SLAM (=mapping), only localization (=localization),"
        + " on another device (=elsewhere) or don't use so map frame is odom frame (=disabled).",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="room_map"),
        description="If localization is used, the name of the map file used for localization.",
    )
    publish_tf_launch_arg = DeclareLaunchArgument(
        "publish_tf",
        default_value=TextSubstitution(text="true"),
        description="If true, the SLAM node will publish the odom to map frame transform",
    )
    namespace_launch_arg = DeclareLaunchArgument(
        "namespace",
        default_value=TextSubstitution(text=""),
        description="Namespace to be used for the nodes and for the /odom and /baselink frames. An empty string means"
        + " that the argument is ignored.",
    )
    start_pose_launch_arg = DeclareLaunchArgument(
        "start_pose",
        default_value=TextSubstitution(text="[0., 0., 0.]"),
        description="2D start pose (x, y, orientation) of the robot in the map.",
    )
    time_interval_launch_arg = DeclareLaunchArgument(
        "time_interval",
        default_value=TextSubstitution(text="1."),
        description="SLAM update interval time (float)"
    )
    slam_mode = LaunchConfiguration("slam_mode")
    map_file = LaunchConfiguration("map_file")
    do_publish_tf = LaunchConfiguration("publish_tf")
    namespace = LaunchConfiguration("namespace")
    start_pose = LaunchConfiguration("start_pose")
    time_interval = LaunchConfiguration("time_interval")

    def setup_slam_node(
        context: LaunchContext,
        slam_mode: LaunchConfiguration,
        map_file: LaunchConfiguration,
        do_publish_tf: LaunchConfiguration,
        namespace: LaunchConfiguration,
        start_pose: LaunchConfiguration,
        time_interval: LaunchConfiguration,
    ):
        do_publish_tf_bool = IfCondition(do_publish_tf).evaluate(context)
        transform_publish_period = 0.05 if do_publish_tf_bool else 0.0
        namespace_str = namespace.perform(context)
        start_pose_str = start_pose.perform(context)
        start_pose_double_array = [float(s) for s in start_pose_str.strip("[]()").split(",")]
        time_interval_float = float(time_interval.perform(context))

        try:
            map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        except KeyError:
            map_files_dir = ""

        common_params = {
            "odom_frame": f"{namespace_str}/odom" if namespace_str else "odom",
            "base_frame": f"{namespace_str}/base_link" if namespace_str else "base_link",
            "transform_publish_period": transform_publish_period,
            "map_start_pose": start_pose_double_array,
        }
        # if slam_mode == "localization"
        slam_node_localization = Node(
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "slam_params.yaml"]),
                {
                    # "map_file_name": PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "map", map_file]),
                    "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
                    "mode": "localization",
                    "do_loop_closing": False,
                    "minimum_time_interval": time_interval_float,
                },
                common_params,
            ],
            package="slam_toolbox",
            executable="localization_slam_toolbox_node",
            name="slam_node",
            namespace=namespace,
            remappings=[("pose", "slam_pose"), ("/scan", "scan")],
            condition=IfCondition(EqualsSubstitution(slam_mode, "localization")),
        )

        # if slam_mode == "mapping"
        slam_node_mapping = Node(
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "slam_params.yaml"]),
                {
                    "mode": "mapping",
                    "do_loop_closing": True,
                    "minimum_time_interval": time_interval_float,
                },
                common_params,
            ],
            package="slam_toolbox",
            executable="sync_slam_toolbox_node",
            name="slam_node",
            namespace=namespace,
            remappings=[("pose", "slam_pose"), ("/scan", "scan")],
            condition=IfCondition(EqualsSubstitution(slam_mode, "mapping")),
            # condition=IfCondition(AndSubstitution(use_mapping, NotSubstitution(use_localization))),
        )

        # if slam_mode == "disabled"
        # TODO: Use the start pose
        transform_slam_disabled = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            # arguments=("--frame-id map --child-frame-id odom").split(" "),
            arguments=["--frame-id", "map", "--child-frame-id", f"{namespace_str}/odom" if namespace_str else "odom"],
            condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
        )
        return [
            slam_node_localization,
            slam_node_mapping,
            transform_slam_disabled,
        ]

    return LaunchDescription(
        [
            slam_mode_launch_arg,
            map_file_launch_arg,
            publish_tf_launch_arg,
            namespace_launch_arg,
            start_pose_launch_arg,
	    time_interval_launch_arg,
            OpaqueFunction(
                function=setup_slam_node,
                args=[slam_mode, map_file, do_publish_tf, namespace, start_pose, time_interval],
            ),
        ]
    )
