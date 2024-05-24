import os
import yaml
import numpy as np
import math
from typing import Tuple, List
from launch import LaunchDescription, LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter, SetRemap, PushRosNamespace
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
    obstacles_file_launch_arg = DeclareLaunchArgument(
        "obstacles_config_file",
        default_value=TextSubstitution(text="obstacle_trajectories.yaml"),
        description="the file that contains the configuration for the obstacle trajectories",
    )
    slam_mode_launch_arg = DeclareLaunchArgument(
        "slam_mode_obs",
        default_value=TextSubstitution(text="localization"),
        choices=["localization", "disabled"],
        description="Whether to use the slam_toolbox in localization mode (=localization) or"
        + " don't use it at all, so map frame == odom frame (=disabled).",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )
    use_ekf_launch_arg = DeclareLaunchArgument(
        "use_ekf_obs",
        default_value=TextSubstitution(text="true"),
        description="Use the extended kalman filter to combined the odometry data and the SLAM localization.",
    )
    do_visualize_launch_arg = DeclareLaunchArgument(
        "do_visualize",
        default_value=TextSubstitution(text="true"),
        description="If true, the planner node ",
    )

    obstacles_config_file = LaunchConfiguration("obstacles_config_file")
    slam_mode = LaunchConfiguration("slam_mode_obs")
    map_file = LaunchConfiguration("map_file")
    use_ekf = LaunchConfiguration("use_ekf_obs")

    log_messages = []

    log_messages.append(
        LogInfo(
            msg="ERROR : Not possible to use the ekf and not use the slam!",
            condition=IfCondition(AndSubstitution(use_ekf, EqualsSubstitution(slam_mode, "disabled"))),
        )
    )

    try:
        map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        log_messages.append(LogInfo(msg="The map is loaded from the path in `ROS_MAP_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_MAP_FILES_DIR` is not set!"))
        map_files_dir = ""

    obstacle_trajectories_node = Node(
        package="foresee_the_unseen",
        executable="obstacle_trajectories_node",
        name="obstacle_trajectories_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "obstacle_trajectories_node.yaml"]),
            {
                "obstacle_config_yaml": PathJoinSubstitution(
                    # [FindPackageShare("foresee_the_unseen"), "resource", obstacles_config_file]
                    ["/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/resource", obstacles_config_file]
                ),
                "do_visualize": True,
            },
        ],
    )

    def get_config_from_yaml(yaml_file: str) -> Tuple[List[str], List[List[float]]]:
        """Loads the content from the yaml file"""
        with open(yaml_file) as f:
            obstacle_config = yaml.safe_load(f)
        obstacle_namespaces = obstacle_config["obstacle_cars"].keys()
        start_poses = [obstacle_config["obstacle_cars"][n]["start_pose"] for n in obstacle_namespaces]
        for start_pose in start_poses:
            assert len(start_pose) == 3 and all(
                [isinstance(s, (int, float)) for s in start_pose]
            ), "start_pose should be 3D array describing the initial [x, y, yaw]"
        assert len(obstacle_namespaces) == len(start_poses), "Should be the same no. of namespaces and start poses"

        return obstacle_namespaces, start_poses

    def slam_localization_based_on_yaml(
        context: LaunchContext, yaml_file: LaunchConfiguration, use_ekf: LaunchConfiguration
    ):
        """Calculate the start pose in the `map` frame, based on the `map` -> `planner` transform and the start
        pose in the planner frame. Initialize a slam node with the right topics and start pose."""
        yaml_filepath_str = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", yaml_file]
        ).perform(context)
        use_ekf_bool = IfCondition(use_ekf).evaluate(context)

        obstacle_namespaces, start_poses = get_config_from_yaml(yaml_filepath_str)

        # Get the map to planner transform
        commonroad_config_yamlpath = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", "commonroad_scenario.yaml"]
        ).perform(context)
        with open(commonroad_config_yamlpath) as f:
            obstacle_config = yaml.safe_load(f)
        map_to_planner_pose = obstacle_config["pose_of_planner_in_map_frame"]
        assert len(map_to_planner_pose) == 3, f"Pose should be [x, y, theta]; {map_to_planner_pose=}"
        x_transf, y_transf, th_transf = map_to_planner_pose
        t_matrix = np.zeros((3, 3))
        r_matrix = np.array([[np.cos(th_transf), -np.sin(th_transf)], [np.sin(th_transf), np.cos(th_transf)]])
        t_matrix[:2, :2] = r_matrix
        t_matrix[2, 2] = 1
        t_matrix[:2, 2] = [x_transf, y_transf]

        # launch the slam node for each robot and publish the start position
        slam_nodes, initialpose_publishers = [], []
        for namespace, start_pose in zip(obstacle_namespaces, start_poses):
            start_xy = np.array([start_pose[0], start_pose[1], 1])
            start_theta = start_pose[2]
            start_xy_tf = t_matrix @ start_xy
            start_theta_tf = start_theta + th_transf
            start_pose_tf = [float(start_xy_tf[0]), float(start_xy_tf[1]), float(start_theta_tf)]
            print(f"========={start_pose_tf}===========")
            slam_node = Node(
                parameters=[
                    PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "config", "slam_params.yaml"]),
                    {
                        "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
                        "use_lifecycle_manager": False,
                        "map_start_pose": start_pose_tf,  # this is the initial pose in the map frame
                        "odom_frame": f"{namespace}/odom",
                        "base_frame": f"{namespace}/base_link",
                        "transform_publish_period": 0.0 if use_ekf_bool else 0.05,
                        "do_loop_closing": False, 
                    },
                ],
                # arguments="--ros-args --log-level debug".split(" "),
                package="slam_toolbox",
                executable="localization_slam_toolbox_node",
                name="slam_node",
                output="screen",
                namespace=namespace,
                remappings=[
                    ("pose", "slam_pose"),
                    # ("/map", "map"),
                    # ("/map_metadata", "map_metadata"),
                    ("/scan", "scan"),
                ],
            )

            slam_nodes.append(slam_node)

        return [
            *slam_nodes,
        ]

    def static_transforms_based_on_yaml(context: LaunchContext, yaml_file: LaunchConfiguration):
        """Just set the `map` -> `odom` transform"""
        yaml_filepath_str = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", yaml_file]
        ).perform(context)
        obstacle_namespaces, start_poses = get_config_from_yaml(yaml_filepath_str)

        # Publish a static transform between the map frame and the odom frame of the obstacle
        static_transforms = []
        for namespace, start_pose in zip(obstacle_namespaces, start_poses):
            static_transform = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=(
                    f"--frame-id map --child-frame-id {namespace}/odom"
                ).split(" "),
                # arguments=(
                #         f"--x {start_pose[0]} --y {start_pose[1]} --z 0 --roll 0 --pitch 0 --yaw {start_pose[2]} "
                #         + f"--frame-id planner --child-frame-id {namespace}/odom"
                #     ).split(" "),
                )
            static_transforms.append(static_transform)

        return static_transforms

    # TODO: if slam_mode == 'localization'
    # check if ekf is used
    # launch the slam_toolbox nodes

    return LaunchDescription(
        [
            # arguments
            slam_mode_launch_arg,
            obstacles_file_launch_arg,
            use_ekf_launch_arg,
            map_file_launch_arg,
            do_visualize_launch_arg,
            # log messages
            *log_messages,
            # nodes
            obstacle_trajectories_node,
            # transforms
            OpaqueFunction(
                function=static_transforms_based_on_yaml,
                args=[obstacles_config_file],
                condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
            ),
            OpaqueFunction(
                function=slam_localization_based_on_yaml,
                args=[obstacles_config_file, use_ekf],
                condition=IfCondition(EqualsSubstitution(slam_mode, "localization")),
            ),
        ]
    )
