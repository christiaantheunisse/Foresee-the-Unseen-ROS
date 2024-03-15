import os
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions import LogInfo
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

############# Control robot with keyboard ###############
### $ ros2 run racing_bot_controller teleop_key_node ###
#########################################################

################ Kill all ROS processes #####################
### $ kill $(ps aux | grep "ros-args" | awk '{print $2}') ###
#############################################################

################ Play Ros Bag ##################
### $ ros2 bag play -l just_on_desk/ --clock ###
################################################

################ Record Ros Bag ##################
### $ ros2 bag record -o some_name /scan /wheel_encoders /imu_data ###
### $ ros2 bag record -o tune_ekf /scan /wheel_encoders /imu_data /visualization/trajectory ###
##################################################

################################################ Run localization on laptop ###############################################################################################################################################
# ros2 launch foresee_the_unseen localization_launch.py slam_params_file:=/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/config/mapper_params_localization.yaml map_file_name:=$ROS_MAP_FILES_DIR/on_the_floor ###
###########################################################################################################################################################################################################################

######## Get tf2_ros tree ##############
### $ ros2 run tf2_tools view_frames ###
########################################

########################################## Store a created map ########################################################
### $ ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: /path/to/file}" ###
#######################################################################################################################

########################################## Bash script to save a map to a certain location ########################
### savemap() { # 1 argument with the file_name -> stored in ~/thesis/map_files
###    filename="$(echo ~/thesis/map_files/$1)"
###    if test -f "$filename.data"; then
###        echo "File already exists! Map is not saved."
###    else
###        echo "Saving map to $filename ..."
###        ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: $filename}"
###    fi
### }
#######################################################################################################################

### If defined, the maps are obtained from `ROS_MAP_FILES_DIR`
### Logs are stored in  `ROS_LOG_FILES_DIR`


# TODO: Separate launch file for the bringing up of the real robot.

# ros2 run  datmo2 datmo_node --ros-args -p use_sim_time:=true -p pub_markers:=true


def generate_launch_description():
    # Values should be `true` or `1` and `false` or `0`
    play_rosbag_launch_arg = DeclareLaunchArgument(
        "play_rosbag",
        default_value=TextSubstitution(text="false"),
        description="If false, bring up the real robot.",
    )
    record_rosbag_launch_arg = DeclareLaunchArgument(
        "record_rosbag",
        default_value=TextSubstitution(text="false"),
        description="If true, launch only the nodes necessary to record the sensor data.",
    )
    use_ekf_launch_arg = DeclareLaunchArgument(
        "use_ekf",
        default_value=TextSubstitution(text="false"),
        description="Use the extended kalman filter for the odometry data."
        "This will also activate and use the imu if available.",
    )
    slam_mode_launch_argument = DeclareLaunchArgument(
        "slam_mode",
        default_value=TextSubstitution(text="disabled"),
        choices=["mapping", "localization", "disabled"],
        description="Which mode of the slam_toolbox to use: SLAM, only localization or don't use.",
    )
    map_file_launch_arg = DeclareLaunchArgument(
        "map_file",
        default_value=TextSubstitution(text="on_the_floor"),
        description="If applicable, the name of the map file used for localization.",
    )
    use_datmo_launch_arg = DeclareLaunchArgument(
        "use_datmo",
        default_value=TextSubstitution(text="false"),
        description="If true, launch the datmo node",
    )
    use_foresee_launch_arg = DeclareLaunchArgument(
        "use_foresee",
        default_value=TextSubstitution(text="false"),
        description="If true, launch the foresee_the_unseen node",
    )
    save_topics_launch_arg = DeclareLaunchArgument(
        "save_topics",
        default_value=TextSubstitution(text="false"),
        description="if true runs the node that saves certain topics as pickle files",
    )
    follow_traject_launch_arg = DeclareLaunchArgument(
        "follow_traject",
        default_value=TextSubstitution(text="false"),
        description="if true follows traject",
    )

    play_rosbag = LaunchConfiguration("play_rosbag")
    record_rosbag = LaunchConfiguration("record_rosbag")
    use_ekf = LaunchConfiguration("use_ekf")
    slam_mode = LaunchConfiguration("slam_mode")
    map_file = LaunchConfiguration("map_file")
    use_datmo = LaunchConfiguration("use_datmo")
    use_foresee = LaunchConfiguration("use_foresee")
    save_topics = LaunchConfiguration("save_topics")
    follow_traject = LaunchConfiguration("follow_traject")

    log_messages = []
    log_messages.append(LogInfo(msg="\n=========================== Launch file logging ==========================="))

    try:
        map_files_dir = os.environ["ROS_MAP_FILES_DIR"]
        log_messages.append(LogInfo(msg="The map is loaded from the path in `ROS_MAP_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_MAP_FILES_DIR` is not set!"))
        map_files_dir = ""

    try:
        log_files_dir = os.environ["ROS_LOG_FILES_DIR"]
        log_messages.append(LogInfo(msg="The log data is stored in the path in `ROS_LOG_FILES_DIR`"))
    except KeyError:
        log_messages.append(LogInfo(msg="`ROS_LOG_FILES_DIR` is not set!"))
        log_files_dir = ""

    log_messages.append(
        LogInfo(msg="Only starting the nodes required to run on a rosbag...", condition=IfCondition(play_rosbag))
    )
    log_messages.append(
        LogInfo(msg="Starting all the nodes necessary to record a rosbag...", condition=IfCondition(record_rosbag))
    )
    log_messages.append(
        LogInfo(
            msg="The extended Kalman filter will be used to estimate the position in the \odom frame...",
            condition=IfCondition(use_ekf),
        )
    )
    log_messages.append(
        LogInfo(
            msg="ERROR! It is undefined to have both `play_rosbag:=True` and `record_rosbag:=True`",
            condition=IfCondition(AndSubstitution(play_rosbag, record_rosbag)),
        )
    )
    log_messages.append(
        LogInfo(
            msg="Datmo node is launched",
            condition=IfCondition(use_datmo),
        )
    )
    log_messages.append(LogInfo(msg="\n================================== END ==================================\n"))

    do_use_sim_time = SetParameter(name="use_sim_time", value=play_rosbag)

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
        # condition=IfCondition(OrSubstitution(NotSubstitution(play_rosbag), record_rosbag)),
    )

    # ONLY used on real robot
    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_imu"), "config", "imu_node.yaml"]),
        ],
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("sllidar_ros2"), "launch", "sllidar_a1_launch.py"])
        ),
        condition=UnlessCondition(play_rosbag),  # not play_rosbag
    )
    trajectory_node = Node(
        package="racing_bot_trajectory_follower",
        executable="trajectory_follower_node",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("racing_bot_trajectory_follower"), "config", "trajectory_follower_node.yaml"]
            ),
        ],
        condition=IfCondition(
            AndSubstitution(NotSubstitution(play_rosbag), follow_traject)
        ),  # not play_rosbag and follow_traject
    )

    # Used ANYWAY
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_odometry"), "config", "odometry_node.yaml"]),
            {"do_broadcast_transform": NotSubstitution(use_ekf)},  # Use either this or ekf transform (set in ekf.yaml)
        ],
    )
    local_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="my_ekf_filter_node",
        output="screen",
        # parameters=[PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "config", "ekf.yaml"])],
        parameters=[PathJoinSubstitution(["/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/config/ekf.yaml"])],
        condition=IfCondition(use_ekf),  # use_ekf
    )
    datmo_node_with_remapping = Node(
        package="datmo",
        executable="datmo_node",
        name="datmo_node",
        condition=IfCondition(AndSubstitution(use_datmo, use_foresee)),
        parameters=[{"min_pub_markers": True}],  # TODO: add .yaml file
        remappings=[("/scan", "/scan/road_env")],
    )
    datmo_node_without_remapping = Node(
        package="datmo",
        executable="datmo_node",
        name="datmo_node",
        condition=IfCondition(AndSubstitution(use_datmo, NotSubstitution(use_foresee))),
        parameters=[{"min_pub_markers": True}],  # TODO: add .yaml file
    )
    # $ ros2 run foresee_the_unseen topics_to_disk_node
    save_topics_node = Node(
        package="foresee_the_unseen",
        executable="topics_to_disk_node",
        name="topics_to_disk_node",
        condition=IfCondition(save_topics),
    )

    global_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # This launch file from the slam_toolbox is downloaded from github, because the version shipped with apt
            # does not provide the same arguments.
            PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "launch", "slam_toolbox", "localization_launch.py"]
            )
        ),
        launch_arguments={
            "slam_params_file": PathJoinSubstitution(
                [FindPackageShare("foresee_the_unseen"), "config", "mapper_params_localization.yaml"]
            ),
            "map_file_name": PathJoinSubstitution([map_files_dir, map_file]),
            # "autostart": False,
        }.items(),
        condition=IfCondition(
            AndSubstitution(NotSubstitution(record_rosbag), EqualsSubstitution(slam_mode, "localization"))
        ),  # (not record_rosbag) and (slam_mode == "localization")
    )

    global_localization_and_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # This launch file from the slam_toolbox is downloaded from github, because the version shipped with apt
            # does not provide the same arguments.
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
            AndSubstitution(NotSubstitution(record_rosbag), EqualsSubstitution(slam_mode, "mapping"))
        ),  # (not record_rosbag) and (slam_mode = "mapping")
    )

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("foresee_the_unseen"), "launch", "planner_launch.py"])
        ),
        launch_arguments={
            "play_rosbag": play_rosbag,
        }.items(),
        condition=IfCondition(use_foresee),  # use_foresee
    )

    # Static transforms
    # $ ros2 run tf2_ros static_transform_publisher --x 0.0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id map --child-frame-id odom
    static_trans_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "odom",
        ],
        condition=IfCondition(EqualsSubstitution(slam_mode, "disabled")),
    )
    # $ ros2 run tf2_ros static_transform_publisher --x 0.085 --y 0 --z 0 --yaw 0 --pitch 0 --roll 3.14 --frame-id laser --child-frame-id base_link
    static_trans_base_link_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.085",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "3.14",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "laser",
        ],
    )
    # The angular velocity and linear acceleration are correctly transformed, but the orientation is wrong
    # $ ros2 run tf2_ros static_transform_publisher --x 0.025 --y -0.038 --z 0 --yaw -1.57080 --pitch 0 --roll 0 --frame-id base_link --child-frame-id imu_link
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.025",
            "--y",
            "-0.038",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "-1.57080",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "imu_link",
        ],
    )

    return LaunchDescription(
        [
            # arguments
            play_rosbag_launch_arg,
            record_rosbag_launch_arg,
            use_ekf_launch_arg,
            slam_mode_launch_argument,
            map_file_launch_arg,
            use_datmo_launch_arg,
            use_foresee_launch_arg,
            save_topics_launch_arg,
            follow_traject_launch_arg,
            # log messages
            *log_messages,
            # parameters
            do_use_sim_time,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            odometry_node,
            controller_node,
            trajectory_node,
            imu_node,
            local_localization_node,
            datmo_node_with_remapping,
            datmo_node_without_remapping,
            save_topics_node,
            # launch files
            lidar_launch,
            global_localization_launch,
            global_localization_and_mapping_launch,
            planner_launch,
            # transforms
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
            static_trans_map_to_odom,
        ]
    )
