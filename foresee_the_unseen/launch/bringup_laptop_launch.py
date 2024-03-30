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


""" 
The bringup file for the laptop. Should (all optional):
    if foresee_the_unseen:
        planner_launch.py -> make a lifecycle node
    if slam_mode == mapping / localization (for the racing bot): 
        appropiate slam_toolbox node
    if obstacles:
        obs_traj_launch.py
            - SLAM for the robots in the appropiate namespaces or static transforms
                - Need to fix the start position. Possible solution: store the transform between the map and planner 
                  frame in a .yaml file and make it accessible to the obs_traj_launch.py and the planner_launch.py.
                  Pass as argument from this file.
            - trajectories for the robots in the appropiate namespaces
"""



def generate_launch_description():
    use_foresee_launch_arg = DeclareLaunchArgument(
        "use_foresee",
        default_value=TextSubstitution(text="false"),
        description="If true, launch the foresee_the_unseen node",
    )
    slam_mode_robot_launch_arg = DeclareLaunchArgument(
        "slam_mode_robot",
        default_value=TextSubstitution(text="disabled"),
        choices=["mapping", "localization", "disabled"],
        description="Which mode of the slam_toolbox to use: SLAM (=mapping), only localization (=localization),"
        + " don't use so map frame is odom frame (=disabled).",
    )
    obstacles_file_launch_arg = DeclareLaunchArgument(
        "obstacles_config_file",
        default_value=TextSubstitution(text="obstacle_trajectories.yaml"),
        description="the file that contains the configuration for the obstacle trajectories",
    )
    use_obstacles_launch_arg = DeclareLaunchArgument(
        "use_obstacles",
        default_value=TextSubstitution(text="true"),
        description="if obstacles are used, which means that the localization and trajectories should be handled",
    )
    slam_mode_obs_launch_arg = DeclareLaunchArgument(
        "slam_mode_obs",
        default_value=TextSubstitution(text="disabled"),
        choices=["localization", "disabled"],
        description="Which mode of the slam_toolbox to use: localization (=localization) or"
        + " don't use, so map frame is odom frame (=disabled).",
    )

    use_foresee = LaunchConfiguration("use_foresee")
    slam_mode_robot = LaunchConfiguration("slam_mode_robot")
    obstacles_config_file = LaunchConfiguration("obstacles_config_file")
    use_obstacles = LaunchConfiguration("use_obstacles")
    slam_mode_obs = LaunchConfiguration("slam_mode_obs")


    return LaunchDescription(
        [
            # arguments
            use_foresee_launch_arg,
            slam_mode_robot_launch_arg,
            obstacles_file_launch_arg,
            use_obstacles_launch_arg,
            slam_mode_obs_launch_arg,
            # nodes
        ]
    )
