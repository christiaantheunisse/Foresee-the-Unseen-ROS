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

        self.declare_parameter("do_visualize_trajectory", False)
        self.declare_parameter("do_store_data", False)
        self.declare_parameter("log_directory", "none")

        self.declare_parameter("control_frequency", 10.0)  # [Hz]
        self.declare_parameter("wheel_base_width", 0.145)  # distance between the wheels on the rear axle [m]
        self.declare_parameter("wheel_base_length", 0.1)  # position of the front axle measured from the rear axle [m]

        # self.declare_parameter("velocity_PID", [5.0, 0.0, 0.0])  # velocity PID constants
        self.declare_parameter("velocity_p", 1.0)  # velocity PID constants
        self.declare_parameter("velocity_pi", [5.0, 0.0])  # velocity PID constants
        self.declare_parameter("steering_k", 0.5)  # steering control gain
        self.declare_parameter("min_corner_radius", 0.5)  # max steering angle [deg]

        self.declare_parameter("do_limit_acceleration", True)
        self.declare_parameter("max_acceleration", 0.5)
        self.declare_parameter("max_velocity", 0.7)

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value
        self.odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value

        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.traj_marker_topic = self.get_parameter("trajectory_marker_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.motor_cmd_topic = self.get_parameter("motor_command_topic").get_parameter_value().string_value

        self.do_visual_traj = self.get_parameter("do_visualize_trajectory").get_parameter_value().bool_value
        self.do_store_data = self.get_parameter("do_store_data").get_parameter_value().bool_value
        self.log_directory = self.get_parameter("log_directory").get_parameter_value().string_value
        self.do_store_data = self.do_store_data if os.path.isdir(self.log_directory) else False
        self.log_directory = create_log_directory(self.log_directory)

        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        self.wheel_base_W = self.get_parameter("wheel_base_width").get_parameter_value().double_value
        self.wheel_base_L = self.get_parameter("wheel_base_length").get_parameter_value().double_value

        # self.velocity_PID = self.get_parameter("velocity_PID").get_parameter_value().double_array_value
        self.velocity_Kp = self.get_parameter("velocity_p").get_parameter_value().double_value
        self.velocity_pi = self.get_parameter("velocity_pi").get_parameter_value().double_array_value
        self.steering_k = self.get_parameter("steering_k").get_parameter_value().double_value
        self.min_corner_radius = self.get_parameter("min_corner_radius").get_parameter_value().double_value

        self.do_lim_acc = self.get_parameter("do_limit_acceleration").get_parameter_value().bool_value
        self.max_abs_acc = self.get_parameter("max_acceleration").get_parameter_value().double_value
        self.max_abs_vel = self.get_parameter("max_velocity").get_parameter_value().double_value

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
        # self.create_subscription(TrajectoryMsg, self.trajectory_topic, self.trajectory_callback, 5)
        # self.create_timer(0.5, self.trajectory_callback)
        self.motor_publisher = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, 10)
        self.traj_marker_publisher = self.create_publisher(MarkerArray, self.traj_marker_topic, 1)

        self.create_timer(1 / self.control_frequency, self.apply_control)
        self.trajectory_callback()

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

    def trajectory_callback(self, msg: TrajectoryMsg = None):
        """Callback for the trajectory topic."""
        # def trajectory_callback(self):
        self.last_target_idx = 0
        # FIXME: Temporary fix
        # fmt: off
        # straight line
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0625, 0.0], [1.1875, 0.0], [1.3125, 0.0], [1.4375, 0.0], [1.55, 0.0], [1.65, 0.0], [1.7375, 0.0], [1.8125, 0.0], [1.875, 0.0], [1.925, 0.0], [1.9625, 0.0], [1.9875, 0.0], [2.0, 0.0], [2.0, 0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5])

        # straight line max speed not reached 
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.3264, 0.0], [0.3778, 0.0], [0.4167, 0.0], [0.4431, 0.0], [0.4569, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0], [0.4583, 0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.2556, 0.2056, 0.1556, 0.1056, 0.0556, 0.0056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5])

        # straight line long 
        # positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0625, 0.0], [1.1875, 0.0], [1.3125, 0.0], [1.4375, 0.0], [1.5625, 0.0], [1.6875, 0.0], [1.8125, 0.0], [1.9375, 0.0], [2.0625, 0.0], [2.1875, 0.0], [2.3125, 0.0], [2.4375, 0.0], [2.5625, 0.0], [2.6875, 0.0], [2.8125, 0.0], [2.9375, 0.0], [3.0625, 0.0], [3.1875, 0.0], [3.3125, 0.0], [3.4375, 0.0], [3.5625, 0.0], [3.6875, 0.0], [3.8125, 0.0], [3.9375, 0.0], [4.0625, 0.0], [4.1875, 0.0], [4.3125, 0.0], [4.4375, 0.0], [4.55, 0.0], [4.65, 0.0], [4.7375, 0.0], [4.8125, 0.0], [4.875, 0.0], [4.925, 0.0], [4.9625, 0.0], [4.9875, 0.0], [5.0, 0.0], [5.0, 0.0]])
        # orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5])

        # left corner many zeros end points
        # positions = np.array([[0.0125, 0.0007], [0.0374, 0.002], [0.0749, 0.0039], [0.1246, 0.0087], [0.1863, 0.0184], [0.2593, 0.0356], [0.3426, 0.0618], [0.4346, 0.1007], [0.5328, 0.1553], [0.634, 0.2284], [0.7256, 0.3133], [0.8062, 0.4087], [0.8732, 0.5141], [0.9265, 0.627], [0.9657, 0.7455], [0.9902, 0.8679], [0.9996, 0.9924], [1.0, 1.1174], [1.0, 1.2424], [1.0, 1.3674], [1.0, 1.4924], [1.0, 1.6174], [1.0, 1.7424], [1.0, 1.8674], [1.0, 1.9924], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        # orientations = np.array([0.0524, 0.0524, 0.075, 0.1251, 0.1876, 0.2626, 0.3502, 0.4502, 0.5628, 0.6878, 0.8129, 0.9379, 1.063, 1.188, 1.3131, 1.4382, 1.5227, 1.5345, 1.5464, 1.5582, 1.5701, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75])

        # left corner 
        # positions = np.array([[0.0125, 0.0007], [0.0374, 0.002], [0.0749, 0.0039], [0.1246, 0.0087], [0.1863, 0.0184], [0.2593, 0.0356], [0.3426, 0.0618], [0.4346, 0.1007], [0.5328, 0.1553], [0.634, 0.2284], [0.7256, 0.3133], [0.8062, 0.4087], [0.8732, 0.5141], [0.9265, 0.627], [0.9657, 0.7455], [0.9902, 0.8679], [0.9989, 0.9799], [1.0, 1.0799], [1.0, 1.1674], [1.0, 1.2424], [1.0, 1.3049], [1.0, 1.3549], [1.0, 1.3924], [1.0, 1.4174], [1.0, 1.4299], [1.0, 1.4299]])
        # orientations = np.array([0.0524, 0.0524, 0.075, 0.1251, 0.1876, 0.2626, 0.3502, 0.4502, 0.5628, 0.6878, 0.8129, 0.9379, 1.063, 1.188, 1.3131, 1.4382, 1.5215, 1.531, 1.5393, 1.5464, 1.5523, 1.557, 1.5606, 1.563, 1.5642, 1.5642])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5])

        # slow left corner
        # positions = np.array([[0.0125, 0.0007], [0.0374, 0.002], [0.0749, 0.0039], [0.1246, 0.0087], [0.1863, 0.0184], [0.2472, 0.0324], [0.3076, 0.0486], [0.366, 0.0708], [0.4235, 0.095], [0.4792, 0.1234], [0.5328, 0.1553], [0.5852, 0.1893], [0.634, 0.2284], [0.6814, 0.2691], [0.7256, 0.3133], [0.7668, 0.3601], [0.8062, 0.4087], [0.8406, 0.4608], [0.8732, 0.5141], [0.9016, 0.5697], [0.9265, 0.627], [0.9489, 0.6853], [0.9657, 0.7455], [0.9804, 0.8062], [0.9902, 0.8679], [0.9963, 0.93], [0.9996, 0.9924], [1.0, 1.0549], [1.0, 1.1174], [1.0, 1.1799], [1.0, 1.2299], [1.0, 1.2674], [1.0, 1.2924], [1.0, 1.3049], [1.0, 1.3049]])
        # orientations = np.array([0.0524, 0.0524, 0.075, 0.1251, 0.1876, 0.2501, 0.3126, 0.3752, 0.4377, 0.5002, 0.5628, 0.6253, 0.6878, 0.7503, 0.8129, 0.8754, 0.9379, 1.0005, 1.063, 1.1255, 1.188, 1.2506, 1.3131, 1.3756, 1.4382, 1.5007, 1.5227, 1.5286, 1.5345, 1.5405, 1.5452, 1.5487, 1.5511, 1.5523, 1.5523])
        # velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        # stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75])
        
        # small corner before 1 meter
        positions = np.array([[0.0125, 0.0], [0.0375, 0.0], [0.075, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.2625, 0.0], [0.35, 0.0], [0.45, 0.0], [0.5625, 0.0], [0.6875, 0.0], [0.8125, 0.0], [0.9375, 0.0], [1.0623, 0.0049], [1.185, 0.0267], [1.3021, 0.0698], [1.4099, 0.1327], [1.5036, 0.2151], [1.5812, 0.3128], [1.6404, 0.4225], [1.6795, 0.5408], [1.6962, 0.6517], [1.7, 0.7516], [1.7, 0.8391], [1.7, 0.9141], [1.7, 0.9766], [1.7, 1.0266], [1.7, 1.0641], [1.7, 1.0891], [1.7, 1.1016], [1.7, 1.1016]])
        orientations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0088, 0.0265, 0.0442, 0.0619, 0.0894, 0.2681, 0.4469, 0.6256, 0.8044, 0.9832, 1.1619, 1.3407, 1.4932, 1.5073, 1.5197, 1.5303, 1.5392, 1.5463, 1.5516, 1.5551, 1.5569, 1.5569])
        velocities = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        stamps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5])
        # fmt: on

        # stamps = stamps + self.get_clock().now().seconds_nanoseconds()[0] + 3  # 5 seconds margin
        self.trajectory = Trajectory(positions, orientations, velocities, stamps)
        self.first_call = True
        self.counter = 0

        """ Original Code """
        # traj_frame = msg.path.header.frame_id
        # positions = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.path.poses])
        # quaternions = [p.pose.orientation for p in msg.path.poses]
        # yaws = [
        #     euler_from_quaternion(*map(lambda attr: getattr(q, attr), ["x", "y", "z", "w"]))[2] for q in quaternions
        # ]
        # velocities = [v.linear.x for v in msg.velocities]
        # trajectory = Trajectory(positions, yaws, velocities)
        # trajectory_transformed = self.transform_trajectory(trajectory, self.map_frame, traj_frame)
        # self.trajectory = trajectory_transformed if trajectory_transformed is not None else self.trajectory

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
        if self.first_call:
            self.counter += 1
            if self.counter > 60:
                self.first_call = False
                self.trajectory.stamps = (
                    self.trajectory.stamps + (np.array(self.get_clock().now().seconds_nanoseconds()) * [1, 1e-9]).sum()
                )
            else:
                return None
        current = np.sum(np.array(self.get_clock().now().seconds_nanoseconds()) * [1, 1e-9])
        mask = self.trajectory.stamps > current
        target_idx = np.argmax(mask) if mask.sum() > 0 else -1  # index of first True and else the last element
        return target_idx

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
        # target_idx = self.get_closest_idx()
        target_idx = self.first_future_idx()
        if target_idx is None:
            self.visualize_trajectory()
            return

        # perpendicular distance error # TODO: Think about which perpendicular
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys
        perp_vec = -np.array([np.cos(self.state.yaw + np.pi / 2), np.sin(self.state.yaw + np.pi / 2)])
        # goal_yaw = self.trajectory.yaws[target_idx]
        # perp_vec = -np.array([np.cos(goal_yaw + np.pi / 2), np.sin(goal_yaw + np.pi / 2)])
        perp_error = dist_vec[target_idx] @ perp_vec

        # calculate the acceleration and subsequently the velocity
        acceleration = self.p_control(self.trajectory.vs[target_idx], self.state.v)
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
            f"v = {self.trajectory.vs[target_idx]:.3f}, v_abs = {self.state.v:.3f},"
            + f" v_norm = {self.output_vel:.3f}, a = {acceleration:.3f}"
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
        """PD controller for the speed."""
        Kp, Ki = self.velocity_pi
        error = target - current
        self.cum_error += error / self.control_frequency
        self.cum_error = np.clip(self.cum_error, -self.max_norm_vel, self.max_norm_vel)
        return Kp * error + Ki * self.cum_error
        # Kp, Ki, Kd = self.velocity_PID
        # error = target - current
        # derror = (error - self.prev_error) * self.control_frequency if self.prev_error is not None else 0
        # self.prev_error = error

        # return Kp * error + Kd * derror

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
