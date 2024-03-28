import os
from launch import LaunchDescription, LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.actions.opaque_function import OpaqueFunction
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
### $ ros2 bag record -o debug_trajectory_mon1 /scan /wheel_encoders /imu_data /trajectory /visualization/trajectory /visualization/planner ###
##################################################

################################################ Run localization on laptop ###############################################################################################################################################
### ros2 launch foresee_the_unseen localization_launch.py slam_params_file:=/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/config/mapper_params_localization.yaml map_file_name:=$ROS_MAP_FILES_DIR/on_the_floor ###
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
    namespace_launch_arg = DeclareLaunchArgument(
        "namespace",
        default_value=TextSubstitution(text="obstacle"),
        description="The namespace used for the maps and all the topics.",
    )
    namespace = LaunchConfiguration("namespace")

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
    )

    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        namespace=namespace,
        parameters=[{
            "left_reverse": True,
            "right_reverse": True,
        }]
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        parameters=[
            PathJoinSubstitution([FindPackageShare("racing_bot_encoder"), "config", "encoder_node.yaml"]),
        ],
        namespace=namespace,
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
        namespace=namespace,
    )

    def namespace_as_string(context: LaunchContext, namespace: LaunchConfiguration):
        namespace_str = namespace.perform(context)

        odometry_node = Node(
            package="racing_bot_odometry",
            executable="odometry_node",
            parameters=[
                PathJoinSubstitution([FindPackageShare("racing_bot_odometry"), "config", "odometry_node.yaml"]),
                {
                    "do_broadcast_transform": True,
                    "odom_frame": f"{namespace_str}/odom",
                    "base_frame": f"{namespace_str}/base_link",
                    "wheel_radius": 0.030,
                    "wheel_base": 0.103,
                },
            ],
            namespace=namespace,
        )

        trajectory_node = Node(
            package="racing_bot_trajectory_follower",
            executable="trajectory_follower_node",
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("racing_bot_trajectory_follower"), "config", "trajectory_follower_node.yaml"]
                ),
                {
                    "follow_mode": "position",
                    "odom_frame": f"{namespace_str}/odom",
                    "wheel_base_width": 0.103,
                    "steering_k": 0.5,
                },
            ],
            namespace=namespace,
        )

        return [
            odometry_node,
            trajectory_node,
        ]

    return LaunchDescription(
        [
            # arguments
            namespace_launch_arg,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            controller_node,
            OpaqueFunction(function=namespace_as_string, args=[namespace]),
        ]
    )
