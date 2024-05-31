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

    obstacles_config_file = LaunchConfiguration("obstacles_config_file")
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
                "do_visualize": do_visualize,
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

    def slam_based_on_yaml(
        context: LaunchContext, yaml_file: LaunchConfiguration, use_ekf: LaunchConfiguration
    ):
        """Calculate the start pose in the `map` frame, based on the `map` -> `planner` transform and the start
        pose in the planner frame. Initialize a slam node with the right topics and start pose."""
        yaml_filepath_str = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", yaml_file]
        ).perform(context)
        use_ekf_bool = IfCondition(use_ekf).evaluate(context)

        obstacle_namespaces, start_poses = get_config_from_yaml(yaml_filepath_str)

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
        for namespace, start_pose_planner in zip(obstacle_namespaces, start_poses):
            # convert start pose in `planner` frame to `map` frame
            start_xy = np.array([start_pose_planner[0], start_pose_planner[1], 1])
            start_theta = start_pose_planner[2]
            start_xy_tf = t_matrix @ start_xy
            start_theta_tf = start_theta + th_transf
            start_pose_map = [float(start_xy_tf[0]), float(start_xy_tf[1]), float(start_theta_tf)]

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
                    "start_pose": str(start_pose_map),
                    "minimum_time_interval": "2.0",
                }.items(),
            )
            slam_launches.append(slam_launch)

            # Publish the start pose in case the SLAM node is run on the robot
            # position_variance = 0.2
            # angle_variance = 0.1
            # covariance_array = [0.0] * 36
            # covariance_array[0] = covariance_array[7] = position_variance
            # covariance_array[35] = angle_variance

            # initialpose_msg = {
            #     "pose": {
            #         "pose": {
            #             "position": {"x": start_pose_map[0], "y": start_pose_map[1]},
            #             "orientation": {
            #                 "z": str(math.sin(start_pose_map[2] / 2)),
            #                 "w": str(math.cos(start_pose_map[2] / 2)),
            #             },
            #         }
            #     },
            # }
            # initialpose_publisher = ExecuteProcess(
            #     cmd=[
            #         # *f"ros2 topic pub -t 1 /{namespace}/initialpose geometry_msgs/msg/PoseWithCovarianceStamped".split(
            #         *f"ros2 topic pub -t 5 -w 1 /{namespace}/initialpose geometry_msgs/msg/PoseWithCovarianceStamped".split(
            #             " "
            #         ),
            #         str(initialpose_msg),
            #         *["--qos-reliability", "reliable"],
            #         *["--qos-durability", "transient_local"],
            #     ],
            #     name="Publish /initialpose",
            # )
            # initialpose_publishers.append(initialpose_publisher)

        return [
            *slam_launches,
            # *initialpose_publishers,
        ]

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
            # OpaqueFunction(
            #     function=static_transforms_based_on_yaml,
            #     args=[obstacles_config_file],
            #     condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
            # ),
            OpaqueFunction(
                function=slam_based_on_yaml,
                args=[obstacles_config_file, use_ekf],
            ),
        ]
    )
