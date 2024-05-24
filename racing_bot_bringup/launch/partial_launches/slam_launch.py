import os
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    TextSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    AndSubstitution,
    NotSubstitution,
)
from launch.conditions import IfCondition
from launch.actions.opaque_function import OpaqueFunction

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    localization_launch_arg = DeclareLaunchArgument(
        "localization",
        default_value=TextSubstitution(text="false"),
        description="If true localization mode is launched",
    )
    mapping_launch_arg = DeclareLaunchArgument(
        "mapping",
        default_value=TextSubstitution(text="false"),
        description="If true mapping mode is launched (localization mode is dominant)",
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
    use_localization = LaunchConfiguration("localization")
    use_mapping = LaunchConfiguration("mapping")
    map_file = LaunchConfiguration("map_file")
    do_publish_tf = LaunchConfiguration("publish_tf")

    def setup_slam_node(
        context: LaunchContext,
        do_publish_tf: LaunchConfiguration,
        use_localization: LaunchConfiguration,
        use_mapping: LaunchConfiguration,
        map_file: LaunchConfiguration,
    ):
        do_publish_tf_bool = IfCondition(do_publish_tf).evaluate(context)
        transform_publish_period = 0.05 if do_publish_tf_bool else 0.0
        
        try:
            map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        except KeyError:
            map_files_dir = ""
        
        # if localization == True
        slam_node_localization = Node(
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "slam_params.yaml"]),
                {
                    # "map_file_name": PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "map", map_file]),
                    "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
                    "transform_publish_period": transform_publish_period,
                    "do_loop_closing": False, 
                },
            ],
            package="slam_toolbox",
            executable="localization_slam_toolbox_node",
            name="slam_node",
            remappings=[("pose", "slam_pose")],
            condition=IfCondition(use_localization),
        )

        # if mapping == True and localization == False
        slam_node_mapping = Node(
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "slam_params.yaml"]),
                {
                    "transform_publish_period": transform_publish_period,
                    "do_loop_closing": True, 
                },
            ],
            package="slam_toolbox",
            executable="sync_slam_toolbox_node",
            name="slam_node",
            remappings=[("pose", "slam_pose")],
            condition=IfCondition(AndSubstitution(use_mapping, NotSubstitution(use_localization))),
        )

        return [
            slam_node_localization,
            slam_node_mapping,
        ]

    # TODO: move all static transform to one file
    # Publish a static transform between the `base_link` (robot) frame and the `laser` (Lidar) frame
    static_trans_laser_baselink = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=(
            "--x 0.085 --y 0 --z 0 --roll 0 --pitch 0 --yaw 3.14 --frame-id base_link --child-frame-id laser"
        ).split(" "),
    )

    return LaunchDescription(
        [
            localization_launch_arg,
            mapping_launch_arg,
            map_file_launch_arg,
            publish_tf_launch_arg,
            static_trans_laser_baselink,
            OpaqueFunction(
                function=setup_slam_node,
                args=[do_publish_tf, use_localization, use_mapping, map_file],
            ),
        ]
    )
