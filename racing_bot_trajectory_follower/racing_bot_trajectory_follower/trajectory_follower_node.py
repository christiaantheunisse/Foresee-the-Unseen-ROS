from rclpy.node import Node
import rclpy
import os
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib as mpl
from dataclasses import dataclass
from typing import Optional, Tuple

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from nav_msgs.msg import Odometry
from std_msgs.msg import Int16MultiArray, ColorRGBA, Header, String
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg

from racing_bot_trajectory_follower.lib.trajectories import *


@dataclass
class State:
    x: Optional[float] = None  # position along x-axis [m]
    y: Optional[float] = None  # position along y-axis [m]
    yaw: Optional[float] = None  # heading (0 along x-axis, ccw positive) [rad]
    v: Optional[float] = None  # velocity [m/s]
    t: Optional[float] = None  # time stamp [s]


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def angle_mod(x):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)
    """
    mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    return mod_angle


def trans_rot_from_euler(t):
    rot_quat = [
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
    ]
    translation = [
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z,
    ]
    rot_scipy = R.from_quat(rot_quat)
    _, _, yaw = rot_scipy.as_euler("xyz")

    return translation, yaw


def matrix_yaw_from_transform(t):
    rot_quat = [
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
    ]
    translation = [
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z,
    ]
    rot_scipy = R.from_quat(rot_quat)
    _, _, yaw = rot_scipy.as_euler("xyz")
    rot_matrix = rot_scipy.as_matrix()
    t_matrix = np.zeros((4, 4))
    t_matrix[:3, :3] = rot_matrix
    t_matrix[:3, 3] = translation
    t_matrix[3, 3] = 1

    return t_matrix, yaw


def make_unique_name(directory, counter: int = 0):
    new_directory = directory if counter == 0 else directory + " (" + str(counter) + ")"
    if os.path.exists(new_directory) and counter < 100:
        return make_unique_name(directory, counter + 1)
    else:
        return new_directory


def create_log_directory(base_dir: str):
    if os.path.exists(base_dir):
        t = time.localtime()
        current_time = time.strftime("%Y-%m-%d %A at %H.%Mu", t)
        log_dir = os.path.join(base_dir, current_time)
        log_dir = make_unique_name(log_dir)
        os.mkdir(log_dir)
        return log_dir
    else:
        raise TypeError(f"The path specified for the log files does not exist: {base_dir}")


class TrajectoryFollowerNode(Node):
    """
    This node implements Stanley steering control and PID speed control which is based on the code from the github
    page PythonRobotics:
        https://github.com/AtsushiSakai/PythonRobotics/blob/bd253e81060c6a11a944016706bd1d87ef72bded/PathTracking/stanley_controller/stanley_controller.py

    A futher explanation of the algorithm is given in the method: `apply_control()`

    The trajectory is obtained from a ROS topic with the custom message type Trajectory. The trajectory and the odometry
    information are both converted to the map frame where the control is applied.
    """

    CONTROL_MODES = ["acceleration", "velocity"]
    FOLLOW_MODES = ["position", "time"]

    def __init__(self):
        super().__init__("trajectory_follower_node")
        self.throttle_duration = 1  # s

        # parameters
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("planner_frame", "planner")
        self.declare_parameter("odom_frame", "odom")

        self.declare_parameter("trajectory_topic", "trajectory")
        self.declare_parameter("trajectory_marker_topic", "visualization/trajectory")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("motor_command_topic", "cmd_motor")

        self.declare_parameter("follow_mode", "time")  # Options: acceleration, velocity
        self.declare_parameter("control_mode", "acceleration")  # Options: acceleration, velocity

        self.declare_parameter("do_visualize_trajectory", False)
        self.declare_parameter("do_store_data", False)
        self.declare_parameter("log_directory", "none")

        self.declare_parameter("control_frequency", 10.0)  # [Hz]
        self.declare_parameter("wheel_base_width", 0.145)  # distance between the wheels on the rear axle [m]
        self.declare_parameter("wheel_base_length", 0.1)  # position of the front axle measured from the rear axle [m]

        # self.declare_parameter("velocity_PID", [5.0, 0.0, 0.0])  # velocity PID constants
        self.declare_parameter("velocity_p", 1.0)  # velocity PID constants
        self.declare_parameter("steering_k", 0.5)  # steering control gain
        self.declare_parameter("min_corner_radius", 0.5)  # max steering angle [deg]

        self.declare_parameter("do_limit_acceleration", True)
        self.declare_parameter("max_acceleration", 0.5)
        self.declare_parameter("max_velocity", 0.7)
        self.declare_parameter("velocity_pwm_rate", 0.6)

        self.declare_parameter("do_follow_preplanned", False)
        self.declare_parameter("traject_file", "none")

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value
        self.odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value

        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.traj_marker_topic = self.get_parameter("trajectory_marker_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.motor_cmd_topic = self.get_parameter("motor_command_topic").get_parameter_value().string_value

        self.follow_mode = self.get_parameter("follow_mode").get_parameter_value().string_value
        assert self.follow_mode in self.FOLLOW_MODES, (
            f"follow mode '{self.follow_mode}' is invalid."
            + f" Should be one of the following options: {self.FOLLOW_MODES}"
        )
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value
        assert self.control_mode in self.CONTROL_MODES, (
            f"control mode '{self.control_mode}' is invalid."
            + f" Should be one of the following options: {self.CONTROL_MODES}"
        )
        self.get_logger().info(f"Control mode: {self.control_mode}")

        self.do_visual_traj = self.get_parameter("do_visualize_trajectory").get_parameter_value().bool_value
        self.do_store_data = self.get_parameter("do_store_data").get_parameter_value().bool_value
        self.log_directory = self.get_parameter("log_directory").get_parameter_value().string_value
        self.do_store_data = self.do_store_data if os.path.isdir(self.log_directory) else False
        if self.do_store_data:
            self.log_directory = create_log_directory(self.log_directory)

        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        self.wheel_base_W = self.get_parameter("wheel_base_width").get_parameter_value().double_value
        self.wheel_base_L = self.get_parameter("wheel_base_length").get_parameter_value().double_value

        self.velocity_Kp = self.get_parameter("velocity_p").get_parameter_value().double_value
        self.steering_k = self.get_parameter("steering_k").get_parameter_value().double_value
        self.min_corner_radius = self.get_parameter("min_corner_radius").get_parameter_value().double_value

        self.do_lim_acc = self.get_parameter("do_limit_acceleration").get_parameter_value().bool_value
        self.max_abs_acc = self.get_parameter("max_acceleration").get_parameter_value().double_value
        self.max_abs_vel = self.get_parameter("max_velocity").get_parameter_value().double_value
        self.vel_pwm_rate = self.get_parameter("velocity_pwm_rate").get_parameter_value().double_value

        self.do_follow_prep = self.get_parameter("do_follow_preplanned").get_parameter_value().bool_value
        self.traject_file = self.get_parameter("traject_file").get_parameter_value().string_value

        self.max_steering_angle, self.max_norm_vel, self.max_norm_acc = self.parameters_ddmr(
            wheel_base_width=self.wheel_base_W,
            wheel_base_length=self.wheel_base_L,
            min_corner_radius=self.min_corner_radius,
            do_limit_acceleration=self.do_lim_acc,
            max_velocity=self.max_abs_vel,
            max_acceleration=self.max_abs_acc,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Variables relevant to the trajectory following
        self.state = State()
        self.prev_error = None
        self.last_target_idx = 0
        self.output_vel = 0
        self.trajectory = None
        self.cum_error = 0

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)
        if not self.do_follow_prep:
            self.create_subscription(TrajectoryMsg, self.trajectory_topic, self.trajectory_callback, 5)
        else:
            with open(self.traject_file, "rb") as f:
                traject_dict = pickle.load(f)
            self.trajectory = Trajectory(
                xys=np.array(traject_dict["positions"]),
                yaws=np.array(traject_dict["orientations"]),
                vs=np.array(traject_dict["velocities"]),
                accs=np.array(traject_dict["accelerations"]),
                stamps=np.array(traject_dict["stamps"])
                + (np.array(self.get_clock().now().seconds_nanoseconds()) * [1, 1e-9]).sum(),
            )
        # self.create_timer(0.5, self.trajectory_callback)
        self.motor_publisher = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, 10)
        self.traj_marker_publisher = self.create_publisher(MarkerArray, self.traj_marker_topic, 1)

        self.create_timer(1 / self.control_frequency, self.apply_control)
        # self.trajectory_callback()

    def parameters_ddmr(
        self,
        wheel_base_width: float,
        wheel_base_length: float,
        min_corner_radius: float,
        do_limit_acceleration: bool = False,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Determine all parameters relevant for the trajectory follower and motor control for a differential-drive
        mobile robot (DDMR)

        Arguments:
            wheel_base_width -- width of the wheel base of the ddmr
            wheel_base_length -- virtual front axle position of the ddmr measured from the rear wheels
            min_corner_radius -- the smallest corner radius it should be able to make

        Returns:
            maximum steering angle, maximum normalized motor velocity, maximum normalized acceleration
        """
        max_steering_angle = np.arctan(wheel_base_length / min_corner_radius)
        # Cannot use full motor velocity for forward driving, because steering is done by varying the speed
        max_norm_motor_vel = 1 / (1 + 0.5 * wheel_base_width / wheel_base_length * np.tan(max_steering_angle))
        if do_limit_acceleration and max_velocity is not None and max_acceleration is not None:
            # limit the maximum normalised acceleration based on the absolute (hardware defined) max velocity
            max_norm_acc = max_acceleration / max_velocity
        else:
            max_norm_acc = None

        self.get_logger().info(
            f"Predefined params: W = {wheel_base_width:.3f} m; L = {wheel_base_length:.3f} m,"
            + f" R_min = {min_corner_radius:.3f} m"
        )
        self.get_logger().info(
            f"Calculated params: delta_max = {max_steering_angle:.3f} rad; V_norm_max = {max_norm_motor_vel:.3f} /s,"
            + f" A_norm_max = {max_norm_acc:.3f} /s2"
        )

        return max_steering_angle, max_norm_motor_vel, max_norm_acc

    def visualize_trajectory(
        self,
        target_idx: Optional[int] = None,
        goal_velocity: Optional[float] = None,
        debug_string: Optional[str] = None,
    ) -> None:
        if not self.do_visual_traj:
            return

        marker_list = []
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)

        if self.trajectory is not None:
            point_list = []
            color_rgba_list = []
            cmap = mpl.cm.get_cmap("cool")
            for (x, y), v in zip(self.trajectory.xys, self.trajectory.vs):
                point_list.append(Point(x=float(x), y=float(y)))
                r, g, b, a = cmap(len(point_list) / len(self.trajectory.xys))
                color_rgba_list.append(ColorRGBA(r=r, g=g, b=b, a=a))
            trajectory_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=0,
                points=point_list,
                colors=color_rgba_list,
                scale=Vector3(x=0.03, y=0.03),
                ns="positions",
            )
            marker_list.append(trajectory_marker)

        if target_idx is not None:
            closest_point_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=0,
                points=[
                    Point(x=float(self.trajectory.xys[target_idx, 0]), y=float(self.trajectory.xys[target_idx, 1]))
                ],
                colors=[ColorRGBA(a=1.0)],
                scale=Vector3(x=0.06, y=0.06),
                ns="closest point",
            )
            marker_list.append(closest_point_marker)

        if self.state.x is not None:
            state_point_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=0,
                points=[Point(x=float(self.state.x), y=float(self.state.y))],
                colors=[ColorRGBA(a=1.0)],
                scale=Vector3(x=0.1, y=0.1),
                ns="state",
            )
            marker_list.append(state_point_marker)

        if goal_velocity is not None:
            goal_velocity_marker = Marker(
                header=header,
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                id=0,
                text=str(str(round(goal_velocity, 3)) + " m/s"),
                color=ColorRGBA(a=1.0),
                scale=Vector3(z=0.2),
                ns="goal velocity",
            )
            marker_list.append(goal_velocity_marker)

        if debug_string is not None:
            debug_text_marker = Marker(
                header=header,
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                id=0,
                text=debug_string,
                color=ColorRGBA(a=1.0),
                scale=Vector3(z=0.2),
                ns="debug",
            )
            marker_list.append(debug_text_marker)

        self.traj_marker_publisher.publish(MarkerArray(markers=marker_list))

    def transform_trajectory(self, trajectory: Trajectory, target_frame: str, source_frame: str):
        """Transforms a trajectory to another frame by looking a ROS transform."""
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform `{source_frame}` to `{target_frame}`: {ex}",
                throttle_duration_sec=self.throttle_duration,
            )
            return None
        translation, rotation = trans_rot_from_euler(t)
        return trajectory.rotate(rotation).translate(translation[:2])

    def trajectory_callback(self, msg: TrajectoryMsg):
        """Callback for the trajectory topic."""
        # def trajectory_callback(self):
        self.last_target_idx = 0
        # FIXME: Temporary fix
        # fmt: off
        # left corner
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0623, 0.0049], [1.185, 0.0267], [1.3021, 0.0698], [1.4099, 0.1327], [1.5036, 0.2151], [1.5812, 0.3128], [1.6404, 0.4225], [1.6795, 0.5408], [1.6962, 0.6517], [1.7, 0.7516], [1.7, 0.8391], [1.7, 0.9141], [1.7, 0.9766], [1.7, 1.0266], [1.7, 1.0641], [1.7, 1.0891], [1.7, 1.1016], [1.7, 1.1016]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0088, 0.0265, 0.0442, 0.0619, 0.0894, 0.2681, 0.4469, 0.6256, 0.8044, 0.9832, 1.1619, 1.3407, 1.4932, 1.5073, 1.5197, 1.5303, 1.5392, 1.5463, 1.5516, 1.5551, 1.5569, 1.5569])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # accelerations = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5])

        # straight with acceleration
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0625, 0.0], [1.1875, 0.0], [1.3125, 0.0], [1.4375, 0.0], [1.5625, 0.0], [1.6875, 0.0], [1.8125, 0.0], [1.9375, 0.0], [2.05, 0.0], [2.15, 0.0], [2.2375, 0.0], [2.3125, 0.0], [2.375, 0.0], [2.425, 0.0], [2.4625, 0.0], [2.4875, 0.0], [2.5, 0.0], [2.5, 0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # accelerations = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5])

        # 4 x 4 box
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0623, 0.0049], [1.185, 0.0267], [1.3021, 0.0698], [1.4099, 0.1327], [1.5036, 0.2151], [1.5812, 0.3128], [1.6404, 0.4225], [1.6795, 0.5408], [1.6972, 0.6642], [1.7, 0.7891], [1.7, 0.9016], [1.7, 1.0016], [1.7, 1.0891], [1.7, 1.1641], [1.7, 1.2266], [1.7, 1.2766], [1.7, 1.3141], [1.7, 1.3391], [1.7, 1.3516], [1.7, 1.3516], [1.7, 1.3641], [1.7, 1.3891], [1.7, 1.4266], [1.7, 1.4766], [1.7, 1.5391], [1.7, 1.6141], [1.7, 1.7016], [1.7, 1.7863], [1.7, 1.8585], [1.7, 1.9182], [1.7, 1.9655], [1.7, 2.0002], [1.7, 2.0224], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0321], [1.7, 2.0446], [1.7, 2.0696], [1.7, 2.1071], [1.7, 2.1571], [1.7, 2.2196], [1.7, 2.2946], [1.7, 2.3821], [1.7, 2.4821], [1.7, 2.5946], [1.7, 2.7196], [1.7, 2.8446], [1.7, 2.9696], [1.6951, 3.0944], [1.6733, 3.2171], [1.6302, 3.3342], [1.5673, 3.442], [1.4849, 3.5357], [1.3872, 3.6133], [1.2775, 3.6725], [1.1592, 3.7116], [1.0358, 3.7293], [0.9109, 3.7321], [0.7984, 3.7321], [0.6984, 3.7321], [0.6109, 3.7321], [0.5359, 3.7321], [0.4734, 3.7321], [0.4234, 3.7321], [0.3859, 3.7321], [0.3609, 3.7321], [0.3484, 3.7321], [0.3484, 3.7321], [0.3359, 3.7321], [0.3109, 3.7321], [0.2734, 3.7321], [0.2234, 3.7321], [0.1609, 3.7321], [0.0859, 3.7321], [-0.0016, 3.7321], [-0.0863, 3.7321], [-0.1585, 3.7321], [-0.2182, 3.7321], [-0.2655, 3.7321], [-0.3002, 3.7321], [-0.3224, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3321, 3.7321], [-0.3446, 3.7321], [-0.3696, 3.7321], [-0.4071, 3.7321], [-0.4571, 3.7321], [-0.5196, 3.7321], [-0.5946, 3.7321], [-0.6821, 3.7321], [-0.7821, 3.7321], [-0.8946, 3.7321], [-1.0196, 3.7321], [-1.1446, 3.7321], [-1.2696, 3.7321], [-1.3944, 3.7272], [-1.5171, 3.7054], [-1.6342, 3.6623], [-1.742, 3.5994], [-1.8357, 3.517], [-1.9133, 3.4193], [-1.9725, 3.3096], [-2.0116, 3.1913], [-2.0293, 3.0679], [-2.0321, 2.9431], [-2.0321, 2.8306], [-2.0321, 2.7306], [-2.0321, 2.6431], [-2.0321, 2.5681], [-2.0321, 2.5056], [-2.0321, 2.4556], [-2.0321, 2.4181], [-2.0321, 2.3931], [-2.0321, 2.3806], [-2.0321, 2.3806], [-2.0321, 2.3681], [-2.0321, 2.3431], [-2.0321, 2.3056], [-2.0321, 2.2556], [-2.0321, 2.1931], [-2.0321, 2.1181], [-2.0321, 2.0306], [-2.0321, 1.9458], [-2.0321, 1.8736], [-2.0321, 1.8139], [-2.0321, 1.7667], [-2.0321, 1.7319], [-2.0321, 1.7097], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.7], [-2.0321, 1.6875], [-2.0321, 1.6625], [-2.0321, 1.625], [-2.0321, 1.575], [-2.0321, 1.5125], [-2.0321, 1.4375], [-2.0321, 1.35], [-2.0321, 1.25], [-2.0321, 1.1375], [-2.0321, 1.0125], [-2.0321, 0.8875], [-2.0321, 0.7625], [-2.0272, 0.6377], [-2.0054, 0.515], [-1.9623, 0.3979], [-1.8994, 0.2901], [-1.817, 0.1964], [-1.7193, 0.1188], [-1.6096, 0.0596], [-1.4913, 0.0205], [-1.3679, 0.0028], [-1.2431, -0.0], [-1.1306, -0.0], [-1.0306, -0.0], [-0.9431, -0.0], [-0.8681, -0.0], [-0.8056, -0.0], [-0.7556, -0.0], [-0.7181, -0.0], [-0.6931, -0.0], [-0.6806, -0.0], [-0.6806, -0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019, 0.0894, 0.2681, 0.4469, 0.6256, 0.8044, 0.9832, 1.1619, 1.3407, 1.5019, 1.5653, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5898, 1.6602, 1.8389, 2.0177, 2.1964, 2.3752, 2.5539, 2.7327, 2.9115, 3.0727, 3.1361, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1606, 3.231, 3.4097, 3.5885, 3.7672, 3.946, 4.1247, 4.3035, 4.4823, 4.6435, 4.7068, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7314, 4.8018, 4.9805, 5.1593, 5.338, 5.5168, 5.6955, 5.8743, 6.0531, 6.2143, 6.2776, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.3389, 0.2889, 0.2389, 0.1889, 0.1389, 0.0889, 0.0389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.3389, 0.2889, 0.2389, 0.1889, 0.1389, 0.0889, 0.0389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.3389, 0.2889, 0.2389, 0.1889, 0.1389, 0.0889, 0.0389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # accelerations = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -0.0444, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -0.0444, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -0.0444, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0, 30.25, 30.5, 30.75, 31.0, 31.25, 31.5, 31.75, 32.0, 32.25, 32.5, 32.75, 33.0, 33.25, 33.5, 33.75, 34.0, 34.25, 34.5, 34.75, 35.0, 35.25, 35.5, 35.75, 36.0, 36.25, 36.5, 36.75, 37.0, 37.25, 37.5, 37.75, 38.0, 38.25, 38.5, 38.75, 39.0, 39.25, 39.5, 39.75, 40.0, 40.25, 40.5, 40.75, 41.0, 41.25, 41.5, 41.75, 42.0, 42.25, 42.5, 42.75, 43.0, 43.25, 43.5, 43.75, 44.0, 44.25, 44.5, 44.75, 45.0, 45.25, 45.5, 45.75, 46.0, 46.25, 46.5, 46.75, 47.0, 47.25, 47.5, 47.75, 48.0, 48.25, 48.5, 48.75, 49.0, 49.25, 49.5, 49.75, 50.0, 50.25, 50.5, 50.75, 51.0, 51.25, 51.5, 51.75, 52.0, 52.25, 52.5, 52.75, 53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75, 55.0, 55.25, 55.5, 55.75, 56.0])

        # 3 x 3 box
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2583, 0.0], [0.3167, 0.0], [0.3625, 0.0], [0.3958, 0.0], [0.4167, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.425, 0.0], [0.4375, 0.0], [0.4625, 0.0], [0.5, 0.0], [0.55, 0.0], [0.6125, 0.0], [0.6875, 0.0], [0.7748, 0.0039], [0.8736, 0.018], [0.9809, 0.0506], [1.0922, 0.1066], [1.1921, 0.1812], [1.2773, 0.2722], [1.3454, 0.3768], [1.3928, 0.4922], [1.4183, 0.6143], [1.425, 0.7266], [1.425, 0.8266], [1.425, 0.9141], [1.425, 0.9891], [1.425, 1.0516], [1.425, 1.1016], [1.425, 1.1391], [1.425, 1.1641], [1.425, 1.1766], [1.425, 1.1766], [1.425, 1.1891], [1.425, 1.2141], [1.425, 1.2516], [1.425, 1.3016], [1.425, 1.3641], [1.425, 1.4391], [1.4211, 1.5264], [1.407, 1.6251], [1.3744, 1.7324], [1.3184, 1.8438], [1.2438, 1.9436], [1.1528, 2.0289], [1.0482, 2.0969], [0.9328, 2.1444], [0.8107, 2.1698], [0.6984, 2.1766], [0.5984, 2.1766], [0.5109, 2.1766], [0.4359, 2.1766], [0.3734, 2.1766], [0.3234, 2.1766], [0.2859, 2.1766], [0.2609, 2.1766], [0.2484, 2.1766], [0.2484, 2.1766], [0.2359, 2.1766], [0.2109, 2.1766], [0.1734, 2.1766], [0.1234, 2.1766], [0.0609, 2.1766], [-0.0099, 2.1766], [-0.0682, 2.1766], [-0.1141, 2.1766], [-0.1474, 2.1766], [-0.1682, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1766, 2.1766], [-0.1891, 2.1766], [-0.2141, 2.1766], [-0.2516, 2.1766], [-0.3016, 2.1766], [-0.3641, 2.1766], [-0.4391, 2.1766], [-0.5264, 2.1726], [-0.6251, 2.1586], [-0.7324, 2.1259], [-0.8438, 2.07], [-0.9436, 1.9954], [-1.0289, 1.9044], [-1.0969, 1.7998], [-1.1444, 1.6844], [-1.1698, 1.5622], [-1.1766, 1.45], [-1.1766, 1.35], [-1.1766, 1.2625], [-1.1766, 1.1875], [-1.1766, 1.125], [-1.1766, 1.075], [-1.1766, 1.0375], [-1.1766, 1.0125], [-1.1766, 1.0], [-1.1766, 1.0], [-1.1766, 0.9875], [-1.1766, 0.9625], [-1.1766, 0.925], [-1.1766, 0.875], [-1.1766, 0.8125], [-1.1766, 0.7375], [-1.1726, 0.6502], [-1.1586, 0.5514], [-1.1259, 0.4441], [-1.07, 0.3328], [-0.9954, 0.2329], [-0.9044, 0.1477], [-0.7998, 0.0796], [-0.6844, 0.0322], [-0.5622, 0.0067], [-0.45, 0.0], [-0.35, 0.0], [-0.2625, 0.0], [-0.1875, 0.0], [-0.125, 0.0], [-0.075, 0.0], [-0.0375, 0.0], [-0.0125, 0.0], [-0.0, 0.0], [-0.0, 0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0317, 0.076, 0.2145, 0.3754, 0.5541, 0.7329, 0.9117, 1.0904, 1.2692, 1.4479, 1.5336, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.6025, 1.6468, 1.7853, 1.9462, 2.1249, 2.3037, 2.4824, 2.6612, 2.84, 3.0187, 3.1044, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1733, 3.2176, 3.3561, 3.517, 3.6957, 3.8745, 4.0532, 4.232, 4.4108, 4.5895, 4.6752, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7124, 4.7441, 4.7884, 4.9269, 5.0878, 5.2665, 5.4453, 5.624, 5.8028, 5.9816, 6.1603, 6.246, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.2833, 0.2333, 0.1833, 0.1333, 0.0833, 0.0333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2833, 0.2333, 0.1833, 0.1333, 0.0833, 0.0333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # accelerations = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1333, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1333, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25, 23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75, 29.0, 29.25, 29.5, 29.75, 30.0, 30.25, 30.5, 30.75, 31.0, 31.25, 31.5, 31.75, 32.0, 32.25, 32.5, 32.75, 33.0, 33.25, 33.5, 33.75, 34.0, 34.25, 34.5, 34.75, 35.0, 35.25, 35.5, 35.75, 36.0, 36.25, 36.5, 36.75, 37.0, 37.25, 37.5])

        # fmt: on

        # self.trajectory = Trajectory(
        #     xys=positions,
        #     yaws=orientations,
        #     vs=velocities,
        #     accs=accelerations,
        #     stamps=stamps,
        # )
        # self.first_call = True
        # self.counter = 0

        """ Original Code """
        traj_frame = msg.path.header.frame_id
        positions = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.path.poses])
        quaternions = [p.pose.orientation for p in msg.path.poses]
        yaws = np.array([
            euler_from_quaternion(*map(lambda attr: getattr(q, attr), ["x", "y", "z", "w"]))[2] for q in quaternions
        ])
        velocities = np.array([v.linear.x for v in msg.velocities])
        accelerations = np.array([a.linear.x for a in msg.accelerations])
        stamps = np.array([pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9 for pose in msg.path.poses])
        trajectory = Trajectory(
            xys=positions,
            yaws=yaws,
            vs=velocities,
            accs=accelerations,
            stamps=stamps,
        )
        trajectory_transformed = self.transform_trajectory(trajectory, self.map_frame, traj_frame)
        self.trajectory = trajectory_transformed if trajectory_transformed is not None else self.trajectory

    def odom_callback(self, msg: Odometry):
        """Callback for the odometry topic. Converts the pose in the odometry to the map frame."""
        position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ]
        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        _, _, yaw = euler_from_quaternion(*quaternion)
        velocity = msg.twist.twist.linear.x
        odom_frame = msg.header.frame_id

        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, odom_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform `{odom_frame}` to `{self.map_frame}`: {ex}",
                throttle_duration_sec=self.throttle_duration,
            )
            return
        # translation, rotation = trans_rot_from_euler(t)
        t_mat, yaw_rot = matrix_yaw_from_transform(t)
        transf_position = (t_mat @ np.array([*position, 0, 1]))[:2]
        transf_yaw = yaw + yaw_rot
        self.prev_state = self.state
        self.state = State(
            *transf_position, transf_yaw, velocity, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )

    def get_closest_idx(self) -> int:
        """Finds the index of the closest point on the trajectory"""
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys
        dist = np.linalg.norm(dist_vec, axis=1)
        target_idx = np.argmin(dist)
        if self.last_target_idx > target_idx:
            target_idx = self.last_target_idx
        else:
            self.last_target_idx = target_idx

        return target_idx

    def first_future_idx(self) -> int:
        """Finds the index of the first future stamp on the trajectory"""
        current = np.sum(np.array(self.get_clock().now().seconds_nanoseconds()) * [1, 1e-9])
        mask = self.trajectory.stamps > current
        target_idx = np.argmax(mask) if mask.sum() > 0 else -1  # index of first True and else the last element
        return target_idx

    def interpolate_target_velocity(self, target_idx: int) -> float:
        """Interpolate the target velocity between the waypoints."""
        if target_idx > 0:
            v_next, v_prev = self.trajectory.vs[target_idx], self.trajectory.vs[target_idx - 1]
            t_next, t_prev = self.trajectory.stamps[target_idx], self.trajectory.stamps[target_idx - 1]
            t_cur = self.state.t
            target_vel = v_prev + (t_cur - t_prev) / (t_next - t_prev) * (v_next - v_prev)
        else:
            target_vel = self.trajectory.vs[target_idx]

        return target_vel

    def apply_control(self):
        """
        Calculates the motor commands based on the trajectory and the position using a Stanley controller for the
        steering angle and a PID controller for the speed.

        The PID controller (actually just a PD controller) is tuned to output the acceleration for a normalized motor
        speed in the range [-1, 1].

        The Stanley controller is developed for Ackermann steering vehicles and not for a differential drive mobile
        robot (DDMR). The position of the front axle is chosen to be on the castor wheel, but this is a arbitrary
        decision. The algorithms tries to position the front axle on the path and to align the (fictional) front wheels
        with the path. The center of the baselink frame (and thus the position the odometry pose is pointing to) is the
        center of the rear axle.
        """
        no_state = self.state.x is None or self.state.y is None or self.state.yaw is None or self.state.v is None
        no_trajectory = self.trajectory is None
        if no_state or no_trajectory:
            if no_state:
                self.get_logger().info(
                    "No state update received, so control is not applied", throttle_duration_sec=self.throttle_duration
                )
            if no_trajectory:
                self.get_logger().info(
                    "No trajectory available, so control is not applied", throttle_duration_sec=self.throttle_duration
                )

            self.visualize_trajectory()
            return

        # calculate target index
        if self.follow_mode == "position":
            target_idx = self.get_closest_idx()
        elif self.follow_mode == "time":
            target_idx = self.first_future_idx()

        if target_idx is None:
            self.visualize_trajectory()
            return

        # perpendicular distance error
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys
        perp_vec = -np.array([np.cos(self.state.yaw + np.pi / 2), np.sin(self.state.yaw + np.pi / 2)])
        # goal_yaw = self.trajectory.yaws[target_idx]
        # perp_vec = -np.array([np.cos(goal_yaw + np.pi / 2), np.sin(goal_yaw + np.pi / 2)])
        perp_error = dist_vec[target_idx] @ perp_vec

        # calculate the acceleration and subsequently the velocity
        target_vel = self.interpolate_target_velocity(target_idx)
        if self.control_mode == "acceleration":  # use the trajectory velocity and acceleration
            acc_traject = self.trajectory.accs[target_idx] / self.vel_pwm_rate  # m/s2 to pwm/s
            acc_error = self.p_control(target_vel, self.state.v)
            acceleration = acc_traject + acc_error
        elif self.control_mode == "velocity":  # only use the trajectory velocity
            acceleration = self.p_control(target_vel, self.state.v)

        # interpolate target velocity
        # if target_idx > 0:
        #     v_next, v_prev = self.trajectory.vs[target_idx], self.trajectory.vs[target_idx - 1]
        #     t_next, t_prev = self.trajectory.stamps[target_idx], self.trajectory.stamps[target_idx - 1]
        #     t_cur = self.state.t
        #     target_vel = v_prev + (t_cur - t_prev) / (t_next - t_prev) * (v_next - v_prev)
        #     acc_error = self.p_control(target_vel, self.state.v)
        # else:
        #     acc_error = 0
        if self.do_lim_acc and not -self.max_norm_acc <= acceleration <= self.max_norm_acc:
            self.get_logger().warn(f"norm_acc > max_norm_acc; {acceleration:.3f} > {self.max_norm_acc:.3f}")
            acceleration = np.clip(acceleration, -self.max_norm_acc, self.max_norm_acc)
        self.output_vel += acceleration / self.control_frequency
        if abs(self.output_vel) > self.max_norm_vel:
            self.output_vel = np.clip(self.output_vel, -self.max_norm_vel, self.max_norm_vel)
            self.get_logger().warn(
                (
                    f"norm_vel > max_norm_vel; {self.output_vel:.3f} > {self.max_norm_vel:.3f}"
                    + f"Speed most likely not reachable: norm_acc = {acceleration}"
                ),
                throttle_duration_sec=self.throttle_duration,
            )
        elif self.output_vel < 0.0:
            self.output_vel = 0.0
            self.get_logger().warn(
                f"norm_vel < 0; {self.output_vel:.3f} < {0.}",
                throttle_duration_sec=self.throttle_duration,
            )

        # calculate the steering angle: ccw positive
        delta = self.stanley_controller(self.trajectory.yaws[target_idx], perp_error)
        if abs(delta) > self.max_steering_angle:
            if self.state.v > 0.01:  # low velocities result in high delta
                self.get_logger().warn(f"delta > max_steering_angle; {delta:.3f} > {self.max_steering_angle:.3f}")
            delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        # convert the steering angle to a speed difference of the wheels: v_delta = (v / 2) * (W / L) * tan(delta)
        # v_delta is proportional to the velocity, so this all works with normalized velocities
        v_delta = self.output_vel / 2 * self.wheel_base_W / self.wheel_base_L * np.tan(delta)

        # directly set the motor speed
        v_left, v_right = self.output_vel - v_delta, self.output_vel + v_delta
        if abs(v_left) > 1 or abs(v_right) > 1:
            v_left, v_right = self.scale_motor_vels(v_left, v_right)
            self.get_logger().error(
                f"SHOULD NOT HAPPEN Maximum motor values exceed due to steering: v_delta = {v_delta}, delta = {delta} rad",
                throttle_duration_sec=self.throttle_duration,
            )

        v_left, v_right = int(255 * v_left), int(255 * v_right)
        v_left = v_left if v_left != 0 else 1
        v_right = v_right if v_right != 0 else 1
        motor_command = Int16MultiArray(data=[v_left, v_right, 0, 0])  # 4 motors are implemented by the hat_node
        self.motor_publisher.publish(motor_command)

        debug_string = (
            f"v_target = {self.trajectory.vs[target_idx]:.3f}, v_current = {self.state.v:.3f},"
            + f" v_norm = {self.output_vel:.3f},\n a_target = {self.trajectory.accs[target_idx]:.3f}"
            + f" acc_current = {acceleration}"
        )
        # debug_string = f"goal_orient = {self.trajectory.yaws[target_idx]}, actual_orient = {self.state.yaw}"
        self.visualize_trajectory(target_idx=target_idx, debug_string=debug_string)

        if self.do_store_data:
            self.store_data_on_disk(target_idx)

    def store_data_on_disk(self, target_idx):
        """Stores the data on disk to investigate the trajectory follower performance."""
        data_dict = {
            "current": self.state,
            "target": State(
                *self.trajectory.xys[target_idx],
                self.trajectory.yaws[target_idx],
                self.trajectory.vs[target_idx],
                self.trajectory.stamps[target_idx],
            ),
        }
        time_stamp = (
            str(self.get_clock().now().seconds_nanoseconds()[0])
            + "_"
            + str(self.get_clock().now().seconds_nanoseconds()[1])
        )
        filename = os.path.join(self.log_directory, f"step {time_stamp}.pickle")
        with open(filename, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def p_control(self, target, current) -> float:
        """P controller for the speed."""
        return self.velocity_Kp * (target - current)

    def stanley_controller(self, goal_yaw, perp_error) -> float:
        """Calculates the steering angle in radians according Stanley steering algorithm"""
        # theta_e corrects the heading error
        theta_e = angle_mod(goal_yaw - self.state.yaw)
        # theta_d corrects the cross track error; np.arctan2(y, x) -> so x and y are inverted (arctan(y / x))
        theta_d = np.arctan2(self.steering_k * perp_error, self.state.v)

        return theta_e + theta_d  # [rad]

    @staticmethod
    def scale_motor_vels(v_left, v_right):
        """Scales the motor velocities to make them stay within the range [-1, 1]."""
        if abs(v_left) > 1 and abs(v_left) > abs(v_right):
            v_right /= abs(v_left)
            v_left /= abs(v_left)
        elif abs(v_right) > 1:
            v_left /= abs(v_right)
            v_right /= abs(v_right)

        return v_left, v_right


def main(args=None):
    rclpy.init(args=args)

    trajectory_follower_node = TrajectoryFollowerNode()

    rclpy.spin(trajectory_follower_node)
    trajectory_follower_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

    node = TrajectoryFollowerNode()

    import matplotlib.pyplot as plt

    # plt.plot(*node.trajectory.xys.T, "k", label="trajectory")
    # plt.plot(*node.trajectory.xys.T, ".k", label="trajectory")

    for (x, y), yaw, v in zip(node.trajectory.xys, node.trajectory.yaws, node.trajectory.vs):
        c = np.clip(v / 0.5, -1, 1)
        color = (c, 0, 1 - c)
        plt.scatter(x, y, color=color)
        orient_point = np.array([np.cos(yaw), np.sin(yaw)]) * 0.02 + [x, y]
        plt.plot(*np.vstack(([x, y], orient_point)).T, color="g")

    plt.gca().set_aspect("equal")
    plt.grid()
    plt.legend()
    plt.show()
