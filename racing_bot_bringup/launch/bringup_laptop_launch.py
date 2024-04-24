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
BRINGUP OBSTACLES:
    $ ros2 launch foresee_the_unseen bringup_obstacle_launch.py lidar_reverse:=true use_ekf:=true namespace:=obstacle_car1 follow_traject:=false

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
        default_value=TextSubstitution(text="true"),
        description="If true, launch the foresee_the_unseen node",
    )
    slam_mode_robot_launch_arg = DeclareLaunchArgument(
        "slam_mode_robot",
        default_value=TextSubstitution(text="localization"),
        choices=["mapping", "localization", "disabled"],
        description="Which mode of the slam_toolbox to use for the robot: SLAM (=mapping), only localization "
        + "(=localization), don't use so map frame is odom frame (=disabled).",
    )
    use_ekf_robot_launch_arg = DeclareLaunchArgument(
        "use_ekf_robot",
        default_value=TextSubstitution(text="false"),
        description="Use the extended kalman filter for the odometry data."
        "This will also activate and use the imu if available.",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )
    use_obstacles_launch_arg = DeclareLaunchArgument(
        "use_obstacles",
        default_value=TextSubstitution(text="true"),
        description="if obstacles are used, which means that the localization and trajectories should be handled",
    )
    obstacles_file_launch_arg = DeclareLaunchArgument(
        "obstacles_config_file",
        default_value=TextSubstitution(text="obstacle_trajectories.yaml"),
        description="the file that contains the configuration for the obstacle trajectories",
    )
    slam_mode_obs_launch_arg = DeclareLaunchArgument(
        "slam_mode_obs",
        default_value=TextSubstitution(text="localization"),
        choices=["localization", "disabled"],
        description="Which mode of the slam_toolbox to use for the obstacle cars: localization (=localization) or"
        + " don't use, so map frame is odom frame (=disabled).",
    )
    use_ekf_obs_launch_arg = DeclareLaunchArgument(
        "use_ekf_obs",
        default_value=TextSubstitution(text="false"),
        description="If the obstacle cars use an extended kalman filter.",
    )

    use_foresee = LaunchConfiguration("use_foresee")
    slam_mode_robot = LaunchConfiguration("slam_mode_robot")
    use_ekf_robot = LaunchConfiguration("use_ekf_robot")
    map_file = LaunchConfiguration("map_file")
    use_obstacles = LaunchConfiguration("use_obstacles")
    obstacles_config_file = LaunchConfiguration("obstacles_config_file")
    slam_mode_obs = LaunchConfiguration("slam_mode_obs")
    use_ekf_obs = LaunchConfiguration("use_ekf_obs")

    log_messages = []
    log_messages.append(LogInfo(msg="\n=========================== Launch file logging ==========================="))

    try:
        map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        log_messages.append(LogInfo(msg="The map is loaded from the path in `ROS_MAP_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_MAP_FILES_DIR` is not set!"))
        map_files_dir = ""

    log_messages.append(LogInfo(msg="\n================================== END ==================================\n"))

    obstacles_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "launch", "partial_launches", "obs_traj_launch.py"]
            )
        ),
        launch_arguments={
            "obstacles_config_file": obstacles_config_file,
            "map_file": map_file,
            "slam_mode": slam_mode_obs,
            "use_ekf": use_ekf_obs,
        }.items(),
        condition=IfCondition(use_obstacles),
    )

    robot_slam_launch = GroupAction(
        actions=[
            SetRemap(src="/pose", dst="/slam_pose"),  # remapping for the nodes in the launch files
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "localization_launch.py"]
                    )
                ),
                launch_arguments={
                    "slam_params_file": PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_localization_ekf.yaml"]
                    ),
                    "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
                }.items(),
                condition=IfCondition(
                    AndSubstitution(use_ekf_robot, EqualsSubstitution(slam_mode_robot, "localization"))
                ),  # use_ekf and slam_mode == "localization"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "online_async_launch.py"]
                    )
                ),
                launch_arguments={
                    "slam_params_file": PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_online_async_ekf.yaml"]
                    ),
                }.items(),
                condition=IfCondition(
                    AndSubstitution(use_ekf_robot, EqualsSubstitution(slam_mode_robot, "mapping"))
                ),  # use_ekf and slam_mode == "mapping"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "localization_launch.py"]
                    )
                ),
                launch_arguments={
                    "slam_params_file": PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_localization.yaml"]
                    ),
                    "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
                }.items(),
                condition=IfCondition(
                    AndSubstitution(NotSubstitution(use_ekf_robot), EqualsSubstitution(slam_mode_robot, "localization"))
                ),  # not use_ekf and slam_mode == "localization"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "online_async_launch.py"]
                    )
                ),
                launch_arguments={
                    "slam_params_file": PathJoinSubstitution(
                        [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_online_async.yaml"]
                    ),
                }.items(),
                condition=IfCondition(
                    AndSubstitution(NotSubstitution(use_ekf_robot), EqualsSubstitution(slam_mode_robot, "mapping"))
                ),  # use_ekf and slam_mode == "mapping"
            ),
        ]
    )

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "launch", "partial_launches", "planner_launch.py"]
            )
        ),
        launch_arguments={
            "play_rosbag": "false",
        }.items(),
        condition=IfCondition(use_foresee),
    )

    return LaunchDescription(
        [
            # arguments
            use_foresee_launch_arg,
            slam_mode_robot_launch_arg,
            use_ekf_robot_launch_arg,
            map_file_launch_arg,
            obstacles_file_launch_arg,
            use_obstacles_launch_arg,
            slam_mode_obs_launch_arg,
            use_ekf_obs_launch_arg,
            # log messages
            *log_messages,
            # launch files
            obstacles_launch,
            robot_slam_launch,
            planner_launch,
        ]
    )
