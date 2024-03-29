import os
import yaml
from launch import LaunchDescription, LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter, SetRemap
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

    obstacles_config_file = LaunchConfiguration("obstacles_config_file")

    obstacle_trajectories_node = Node(
        package="foresee_the_unseen",
        executable="obstacle_trajectories_node",
        name="obstacle_trajectories_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "obstacle_trajectories_node.yaml"]),
            {
                "obstacle_config_yaml": PathJoinSubstitution(
                    [FindPackageShare("foresee_the_unseen"), "resource", obstacles_config_file]
                )
            },
        ],
    )

    def static_transforms_based_on_yaml(context: LaunchContext, yaml_file: LaunchConfiguration):
        yaml_filepath_str = PathJoinSubstitution(
            [FindPackageShare("foresee_the_unseen"), "resource", yaml_file]
        ).perform(context)
        with open(yaml_filepath_str) as f:
            obstacle_config = yaml.safe_load(f)
        obstacle_namespaces = obstacle_config["obstacle_cars"].keys()
        start_poses = [obstacle_config["obstacle_cars"][n]["start_pose"] for n in obstacle_namespaces]

        # Publish a static transform between the map frame and the odom frame of the obstacle
        static_transforms = []
        for namespace, start_pose in zip(obstacle_namespaces, start_poses, strict=True):
            assert len(start_pose) == 3 and all(
                [isinstance(s, (int, float)) for s in start_pose]
            ), "start_pose should be 3D array describing the initial [x, y, yaw]"
            static_transform = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=[
                    "--x",
                    str(start_pose[0]),
                    "--y",
                    str(start_pose[1]),
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    str(start_pose[2]),
                    "--frame-id",
                    "planner",
                    "--child-frame-id",
                    f"{namespace}/odom",
                ],
            )
            static_transforms.append(static_transform)
        
        return static_transforms

    return LaunchDescription(
        [
            # arguments
            obstacles_file_launch_arg,
            # nodes
            obstacle_trajectories_node,
            # transforms
            OpaqueFunction(
                function=static_transforms_based_on_yaml,
                args=[obstacles_config_file],
            ),
        ]
    )
