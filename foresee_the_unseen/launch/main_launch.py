import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.substitutions import TextSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

############## Control robot with keyboard ################################
### $ ros2 run racing_bot_controller teleop_key_node ###
#########################################################
 
def generate_launch_description():
    # Values should be `true` or `1` and `false` or `0`
    use_lidar_launch_arg = DeclareLaunchArgument("use_lidar", default_value=TextSubstitution(text="true"))
    use_rosbag_launch_arg = DeclareLaunchArgument("use_rosbag", default_value=TextSubstitution(text="false"))

    # To start the `pigpiod package`, necessary for I2C
    start_pigpiod = ExecuteProcess(
        cmd=["sudo", "pigpiod"],
        name="Start pigpiod",
        condition=IfCondition(LaunchConfiguration("use_rosbag")),
    )

    # ONLY used on real robot
    hat_node = Node(
        package="racing_bot_hat",
        executable="hat_node",
        condition=IfCondition(LaunchConfiguration("use_rosbag")),
    )
    encoder_node = Node(
        package="racing_bot_encoder",
        executable="encoder_node",
        condition=IfCondition(LaunchConfiguration("use_rosbag")),
    )
    controller_node = Node(
        package="racing_bot_controller",
        executable="controller_node",
        condition=IfCondition(LaunchConfiguration("use_rosbag")),
    )
    imu_node = Node(
        package="racing_bot_imu",
        executable="imu_node",
        condition=IfCondition(LaunchConfiguration("use_rosbag")),
    )
    # Launch lidar nodes if `use_lidar` == True
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("sllidar_ros2"), "launch/sllidar_a1_launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("use_lidar")),
    )

    # Used ANYWAY
    odometry_node = Node(
        package="racing_bot_odometry",
        executable="odometry_node",
        ros_arguments=["-p", "do_broadcast_transform:=false", # Use either this or ekf transform (set in ekf.yaml)
                    #    "-p", "pose_variances:=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                    #    "-p", "twist_variances:=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                       "-p", f"pose_variances:={[0.0001]*6}",
                       "-p", f"twist_variances:={[0.0001]*6}",
                       ],  
        output="screen",
    )
    visualization_node = Node(
        package="foresee_the_unseen",
        executable="visualization_node",
    )
    # Extended Kalman Filter: https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
    # pkg_foresee_the_unseen = FindPackageShare("foresee_the_unseen")
    # path_ekf_yaml = PathJoinSubstitution([pkg_foresee_the_unseen, "config", "ekf.yaml"])
    path_ekf_yaml = "/home/ubuntu/thesis_ws/src/foresee_the_unseen/config/ekf.yaml" # don't need to build
    robot_localization_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="my_ekf_filter_node",
        output="screen",
        parameters=[path_ekf_yaml],
    )

    # Static transforms
    # $ ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /odom --child-frame-id /map
    static_trans_odom_to_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "odom",
            "--child-frame-id", "map",
        ],
    )
    # $ ros2 run tf2_ros static_transform_publisher --x -0.1 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /laser --child-frame-id /base_link
    static_trans_base_link_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.85",
            "--y", "0",
            "--z", "0",
            "--roll", "3.14",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "laser",
        ],
    )
    # $ ros2 run tf2_ros static_transform_publisher --x -0.035 --y 0.04 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /base_link --child-frame-id /imu_link
    static_trans_base_link_to_imu_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.035",
            "--y", "0.04",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "imu_link",
        ],
    )

    return LaunchDescription(
        [
            # arguments
            use_lidar_launch_arg,
            # commands
            start_pigpiod,
            # nodes
            hat_node,
            encoder_node,
            odometry_node,
            controller_node,
            visualization_node,
            imu_node,
            robot_localization_node,
            # launch files
            lidar_launch,
            # transforms
            static_trans_odom_to_map,
            static_trans_base_link_to_laser,
            static_trans_base_link_to_imu_link,
        ]
    )
