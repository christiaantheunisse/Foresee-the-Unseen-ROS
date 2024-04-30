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

###### launch with a rosbag
# ros2 launch racing_bot_bringup bringup_laptop_launch.py slam_mode_robot:=mapping play_rosbag:=true use_foresee:=false


def generate_launch_description():
    use_foresee_launch_arg = DeclareLaunchArgument(
        "use_foresee",
        default_value=TextSubstitution(text="true"),
        description="If true, launch the foresee_the_unseen node",
    )
    slam_mode_robot_launch_arg = DeclareLaunchArgument(
        "slam_mode_robot",
        default_value=TextSubstitution(text="mapping"),
        choices=["mapping", "localization", "disabled"],
        description="Which mode of the slam_toolbox to use for the robot: SLAM (=mapping), only localization "
        + "(=localization), don't use so map frame is odom frame (=disabled).",
    )
    use_ekf_robot_launch_arg = DeclareLaunchArgument(
        "use_ekf_robot",
        default_value=TextSubstitution(text="true"),
        description="If the extended kalman filter is used on the robot.",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )

    # Arguments regarding the obstacles -> not fully working yet
    use_obstacles_launch_arg = DeclareLaunchArgument(
        "use_obstacles",
        default_value=TextSubstitution(text="false"),
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

    # Rosbag argument
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="Run some robot nodes when playing rosbag data and use the simulation (rosbag) time",
    )
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "rosbag_file",
        default_value=TextSubstitution(text="none"),
        description="If other than none, the rosbag file to use from ROS_BAG_FILES_DIR",
    )
    store_topics_launch_arg = DeclareLaunchArgument(
        "store_topics",
        default_value=TextSubstitution(text="false"),
        description="If true, save certain topics to disk",
    )

    try:
        bag_files_dir = os.environ["ROS_BAG_FILES_DIR"]
    except KeyError:
        bag_files_dir = ""

    use_foresee = LaunchConfiguration("use_foresee")
    slam_mode_robot = LaunchConfiguration("slam_mode_robot")
    use_ekf_robot = LaunchConfiguration("use_ekf_robot")
    map_file = LaunchConfiguration("map_file")
    use_obstacles = LaunchConfiguration("use_obstacles")
    obstacles_config_file = LaunchConfiguration("obstacles_config_file")
    slam_mode_obs = LaunchConfiguration("slam_mode_obs")
    use_ekf_obs = LaunchConfiguration("use_ekf_obs")
    play_rosbag = LaunchConfiguration("play_rosbag")
    rosbag_file = LaunchConfiguration("rosbag_file")
    store_topics = LaunchConfiguration("store_topics")

    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    # TODO: The obstacle part is not working yet
    obstacles_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "obs_traj_launch.py"]
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

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "slam_launch.py"]
            )
        ),
        launch_arguments={
            "localization": EqualsSubstitution(slam_mode_robot, "localization"),
            "mapping": EqualsSubstitution(slam_mode_robot, "mapping"),
            "map_file": map_file,
            "publish_tf": NotSubstitution(use_ekf_robot),
        }.items(),
    )

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "planner_launch.py"]
            )
        ),
        condition=IfCondition(use_foresee),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", "/home/christiaan/.rviz2/raspberry_pi.rviz"],
    )

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "launch", "bringup_robot_launch.py"])
        ),
        launch_arguments={
            "use_ekf": "true",
            "slam_mode": "elsewhere",
            "play_rosbag": "true",
        }.items(),
        condition=IfCondition(play_rosbag),
    )

    rosbag_player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            PathJoinSubstitution([bag_files_dir, rosbag_file]),
            "--clock",
        ],
        # play_rosbag and rosbag_file != "none"
        condition=IfCondition(AndSubstitution(play_rosbag, NotSubstitution(EqualsSubstitution(rosbag_file, "none")))),
    )

    store_topics_node = Node(
        package="foresee_the_unseen",
        executable="topics_to_disk_node",
        condition=IfCondition(store_topics),
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
            play_rosbag_launch_arg,
            store_topics_launch_arg,
            # parameters
            do_use_sim_time,
            # launch files
            obstacles_launch,
            slam_launch,
            planner_launch,
            robot_launch,
            # nodes
            rviz,
            store_topics_node,
            # commands
            rosbag_player,
        ]
    )
