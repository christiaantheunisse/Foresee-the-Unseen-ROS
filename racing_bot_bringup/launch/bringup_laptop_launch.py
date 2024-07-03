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
        choices=["mapping", "localization", "disabled", "elsewhere"],
        description="Which mode of the slam_toolbox to use for the robot: SLAM (=mapping), only localization "
        + "(=localization), don't use so map frame is odom frame (=disabled).",
    )
    use_ekf_robot_launch_arg = DeclareLaunchArgument(
        "use_ekf_robot",
        default_value=TextSubstitution(text="true"),
        description="If the extended kalman filter is used on the robot.",
    )
    use_obstacles_launch_arg = DeclareLaunchArgument(
        "use_obstacles",
        default_value=TextSubstitution(text="false"),
        description="if obstacles are used, which means that the localization and trajectories should be handled",
    )
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="Run some robot nodes when playing rosbag data and use the simulation (rosbag) time",
    )
    rosbag_file_launch_arg = DeclareLaunchArgument(
        "rosbag_file",
        default_value=TextSubstitution(text="none"),
        description="If other than none, the rosbag file to use from ROS_BAG_FILES_DIR",
    )
    store_topics_launch_arg = DeclareLaunchArgument(
        "store_topics",
        default_value=TextSubstitution(text="false"),
        description="If true, save certain topics to disk",
    )
    rviz_file_launch_arg = DeclareLaunchArgument(
        "rviz_file",
        default_value="medium_mode.rviz",
        description=".rviz file to use from the package directory rviz.",
    )
    logging_launch_arg = DeclareLaunchArgument(
        "logging",
        default_value="true",
        description="log the necessary topics for the experiments",
    )
    do_simulate_launch_arg = DeclareLaunchArgument(
        "do_simulate", default_value="false", description="Simulate the odometry and lidar from the robot vehicle"
    )

    try:
        bag_files_dir = os.environ["ROS_BAG_FILES_DIR"]
    except KeyError:
        bag_files_dir = ""

    use_foresee = LaunchConfiguration("use_foresee")
    slam_mode_robot = LaunchConfiguration("slam_mode_robot")
    use_ekf_robot = LaunchConfiguration("use_ekf_robot")
    use_obstacles = LaunchConfiguration("use_obstacles")
    play_rosbag = LaunchConfiguration("play_rosbag")
    rosbag_file = LaunchConfiguration("rosbag_file")
    store_topics = LaunchConfiguration("store_topics")
    rviz_file = LaunchConfiguration("rviz_file")
    do_log = LaunchConfiguration("logging")
    do_simulate = LaunchConfiguration("do_simulate")

    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    obstacles_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "obs_traj_launch.py"]
            )
        ),
        condition=IfCondition(use_obstacles),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "slam_launch.py"]
            )
        ),
        launch_arguments={
            "slam_mode": slam_mode_robot,
            "publish_tf": NotSubstitution(use_ekf_robot),
            "time_interval": "0.5",
            "namespace": "",
            "start_pose": "[0, 0, 0]",
        }.items(),
        condition=IfCondition(NotSubstitution(EqualsSubstitution(slam_mode_robot, "disabled"))),
    )

    planner_launch = GroupAction(
        actions=[
            SetRemap(src="/scan", dst="/scan/simulated", condition=IfCondition(use_obstacles)),  # FIXME: not working
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("racing_bot_bringup"), "launch", "partial_launches", "planner_launch.py"]
                    )
                ),
            ),
        ],
        condition=IfCondition(use_foresee),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        # arguments=["-d", "/home/christiaan/.rviz2/raspberry_pi.rviz"],
        # arguments=["-d", "/home/christiaan/thesis/robot_ws/src/racing_bot_bringup/rviz/full_mode.rviz"],
        # arguments=["-d", PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "rviz", rviz_file])],
        arguments=[
            "-d",
            PathJoinSubstitution(["/home/christiaan/thesis/robot_ws/src/racing_bot_bringup/rviz/", rviz_file]),
        ],
    )

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("racing_bot_bringup"), "launch", "bringup_robot_launch.py"])
        ),
        launch_arguments={
            "use_ekf": "true",
            "slam_mode": "elsewhere",
            "play_rosbag": "true",
            "follow_traject": "false",
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

    logging_node = Node(
        package="foresee_the_unseen",
        executable="logging_node",
        condition=IfCondition(do_log),
    )

    simulate_action = GroupAction(
        actions=[
            Node(
                package="racing_bot_simulation",
                executable="scan_sensor_node",
            ),
            Node(
                package="racing_bot_simulation",
                executable="odometry_sensor_node",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=(
                    "--x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 3.14 --frame-id base_link --child-frame-id laser"
                ).split(" "),
            ),
        ],
        condition=IfCondition(do_simulate),
    )

    return LaunchDescription(
        [
            # arguments
            use_foresee_launch_arg,
            slam_mode_robot_launch_arg,
            use_ekf_robot_launch_arg,
            use_obstacles_launch_arg,
            play_rosbag_launch_arg,
            rosbag_file_launch_arg,
            store_topics_launch_arg,
            rviz_file_launch_arg,
            logging_launch_arg,
            do_simulate_launch_arg,
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
            logging_node,
            # scan_sensor_node,
            # odom_sensor_node,
            # static_trans_base_link_to_laser,
            simulate_action,
            # commands
            rosbag_player,
        ]
    )
