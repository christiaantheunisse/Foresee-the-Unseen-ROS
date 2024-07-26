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
    slam_mode_launch_arg = DeclareLaunchArgument(
        "slam_mode_obs",
        default_value=TextSubstitution(text="elsewhere"),
        choices=["mapping", "localization", "disabled", "elsewhere"],
        description="Which mode of the slam_toolbox to use for the robot: SLAM (=mapping), only localization "
        + "(=localization), don't use so map frame == odom frame (=disabled).",
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

    slam_mode = LaunchConfiguration("slam_mode_obs")
    map_file = LaunchConfiguration("map_file")
    use_ekf = LaunchConfiguration("use_ekf_obs")
    do_visualize = LaunchConfiguration("do_visualize")

    log_messages = []

    log_messages.append(
        LogInfo(
            msg="ERROR : Not possible to use the ekf and not use the slam!",
            condition=IfCondition(AndSubstitution(use_ekf, EqualsSubstitution(slam_mode, "disabled"))),
        )
    )

    obstacle_trajectories_node = Node(
        package="foresee_the_unseen",
        executable="obstacle_trajectories_node",
        name="obstacle_trajectories_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "obstacle_trajectories_node.yaml"]),
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "resource", "ros_params_scenario.yaml"]),
            {
                "obstacle_config_yaml": PathJoinSubstitution(
                    [FindPackageShare("foresee_the_unseen"), "resource", "commonroad_scenario.yaml"]
                    # ["/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/resource", commonroad_config_file]
                ),
                "do_visualize": do_visualize,
            },
        ],
    )

    # TODO: make optional with a launch argument
    scan_simulate_node = Node(
        package="racing_bot_scan_sim",
        executable="scan_simulate_node",
        name="scan_simulate_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_scan_sim"), "config", "scan_simulate_node.yaml"]),
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "resource", "ros_params_scenario.yaml"]),
        ]
    )

    # def get_config_from_yaml(yaml_file: str) -> Tuple[List[str], List[List[float]]]:
    def get_config_from_yaml(yaml_file: str) -> List[str]:
        """Loads the content from the yaml file"""
        with open(yaml_file) as f:
            obstacle_config = yaml.safe_load(f)["obstacle_trajectories"]
        obstacle_namespaces = obstacle_config["obstacle_cars"].keys()
        # start_poses = [obstacle_config["obstacle_cars"][n]["start_pose"] for n in obstacle_namespaces]
        # for start_pose in start_poses:
        #     assert len(start_pose) == 3 and all(
        #         [isinstance(s, (int, float)) for s in start_pose]
        #     ), "start_pose should be 3D array describing the initial [x, y, yaw]"
        # assert len(obstacle_namespaces) == len(start_poses), "Should be the same no. of namespaces and start poses"

        # return obstacle_namespaces, start_poses
        return obstacle_namespaces

    def slam_based_on_yaml(
        context: LaunchContext, use_ekf: LaunchConfiguration
    ):
        """Calculate the start pose in the `map` frame, based on the `map` -> `planner` transform and the start
        pose in the planner frame. Initialize a slam node with the right topics and start pose."""
        yaml_file = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", "commonroad_scenario.yaml"]
        )
        yaml_filepath_str = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", yaml_file]
        ).perform(context)
        use_ekf_bool = IfCondition(use_ekf).evaluate(context)

        # obstacle_namespaces, start_poses = get_config_from_yaml(yaml_filepath_str)
        obstacle_namespaces = get_config_from_yaml(yaml_filepath_str)

        # Get the `planner` to `map` frame transform
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
        slam_launches, initialpose_publishers = [], []
        # for namespace, start_pose_planner in zip(obstacle_namespaces, start_poses):
        if len(obstacle_namespaces) > 2:
            obstacle_namespaces = list(obstacle_namespaces)[:2]
        for namespace in obstacle_namespaces:
            # convert start pose in `planner` frame to `map` frame
            # start_xy = np.array([start_pose_planner[0], start_pose_planner[1], 1])
            # start_theta = start_pose_planner[2]
            # start_xy_tf = t_matrix @ start_xy
            # start_theta_tf = start_theta + th_transf
            # start_pose_map = [float(start_xy_tf[0]), float(start_xy_tf[1]), float(start_theta_tf)]
            # print(str(start_pose_map))
            slam_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "slam_launch.py"]
                    )
                ),
                launch_arguments={
                    "slam_mode": slam_mode,
                    "map_file": map_file,
                    "publish_tf": NotSubstitution(use_ekf),
                    "namespace": namespace,
                    # "start_pose": str(start_pose_map),
                    "time_interval": "2.0",
                }.items(),
            )
            slam_launches.append(slam_launch)

        return [
            *slam_launches,
        ]

    return LaunchDescription(
        [
            # arguments
            slam_mode_launch_arg,
            use_ekf_launch_arg,
            map_file_launch_arg,
            do_visualize_launch_arg,
            # log messages
            *log_messages,
            # nodes
            obstacle_trajectories_node,
            scan_simulate_node,
            OpaqueFunction(
                function=slam_based_on_yaml,
                args=[use_ekf],
                condition=IfCondition(NotSubstitution(EqualsSubstitution(slam_mode, "elsewhere"))),
            ),
        ]
    )
