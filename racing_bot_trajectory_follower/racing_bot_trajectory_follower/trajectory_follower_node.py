from rclpy.node import Node
import rclpy
import os
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib as mpl
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from nav_msgs.msg import Odometry
from std_msgs.msg import Int16MultiArray, ColorRGBA, Header, String
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg

# from racing_bot_trajectory_follower.lib.trajectories import Trajectory


@dataclass
class State:
    x: float  # position along x-axis [m]
    y: float  # position along y-axis [m]
    yaw: float  # heading (0 along x-axis, ccw positive) [rad]
    v_lin: float  # linear velocity [m/s]
    v_ang: float  # angular velocity [rad/s]
    t: float  # time stamp [s]


@dataclass
class Trajectory:
    xys: np.ndarray  # xy-positions [m] [[x, y], ...]
    yaws: np.ndarray  # headings [rad] [yaw, ...]
    vs: np.ndarray  # velocties [m/s] [v, ...]
    accs: np.ndarray  # accelerations [m/s2] [a, ...]
    stamps: np.ndarray  # time stamps [s] [t, ...]

    def translate(self, translation: Union[np.ndarray, List] = [0, 0]):
        """In place translation"""
        # new = copy.deepcopy(self)
        self.xys = self.xys + translation

        return self

    def rotate(self, theta: float = 0):
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.xys = (rot_matrix @ self.xys.T).T
        self.yaws = self.yaws + theta

        return self


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
        self.declare_parameter("linear_velocity_Kp", 1.0)  # velocity PID constants
        self.declare_parameter("angular_velocity_Kp", 0.5)  # velocity PID constants
        self.declare_parameter("angular_velocity_Kd", 0.5)  # velocity PID constants
        self.declare_parameter("steering_k", 0.5)  # steering control gain
        self.declare_parameter("min_corner_radius", 0.5)  # max steering angle [deg]
        self.declare_parameter(
            "goal_distance_lead", 0.15
        )  # distance the goal point is ahead of the current position [m]

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

        self.lin_velocity_Kp = self.get_parameter("linear_velocity_Kp").get_parameter_value().double_value
        self.ang_velocity_Kp = self.get_parameter("angular_velocity_Kp").get_parameter_value().double_value
        self.ang_velocity_Kd = self.get_parameter("angular_velocity_Kd").get_parameter_value().double_value
        self.steering_k = self.get_parameter("steering_k").get_parameter_value().double_value
        self.min_corner_radius = self.get_parameter("min_corner_radius").get_parameter_value().double_value
        self.goal_distance_lead = self.get_parameter("goal_distance_lead").get_parameter_value().double_value

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
        self.state: Optional[State] = None
        self.last_target_idx: int = 0
        self.output_vel: float = 0.0
        self.trajectory: Optional[Trajectory] = None
        self.target_angular_velocity: Optional[float] = None
        self.output_v_delta: float = 0.0
        self.prev_ang_error: Optional[float] = None

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
        self.motor_publisher = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, 10)
        self.traj_marker_publisher = self.create_publisher(MarkerArray, self.traj_marker_topic, 1)

        self.create_timer(1 / self.control_frequency, self.apply_control)

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
            wheel_base_width -- width of the wheel base of the ddmr [m]
            wheel_base_length -- virtual front axle position of the ddmr measured from the rear wheels [m]
            min_corner_radius -- the smallest corner radius it should be able to make [m]
            do_limit_acceleration -- if the acceleration of the robot is limited
            max_velocity -- the maximum reachable velocity of the robot. So the absolute velocity at a normalised
                velocity of 1. [m/s]
            max_acceleration -- the allowed maximum acceleration of the robot [m/s^2]

        Returns:
            maximum steering angle, maximum normalized motor velocity, maximum normalized acceleration
        """
        max_steering_angle = np.arctan(wheel_base_length / min_corner_radius)
        # Cannot use full motor velocity for forward driving, because steering is done by varying the speed
        max_norm_motor_vel = 1 / (1 + 0.5 * wheel_base_width / wheel_base_length * np.tan(max_steering_angle))
        if do_limit_acceleration and max_velocity is not None and max_acceleration is not None:
            # limit the maximum normalised acceleration based on the absolute (hardware defined) max velocity
            max_norm_acc = max_acceleration / max_velocity
        elif do_limit_acceleration:
            self.get_logger().error(
                (
                    f"In order to enable `do_limit_acceleration` the `max_velocity` and `max_acceleration` should",
                    " be specified. The maximum acceleration can be chosen, but the maximum velocity should be set",
                    "to the maximum reachable velocity of the robot.",
                )
            )
            max_norm_acc = np.inf
        else:
            max_norm_acc = np.inf

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
        target_idx_p: Optional[int] = None,
        target_idx_v: Optional[int] = None,
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

        if target_idx_p is not None or target_idx_v is not None:
            points, colors = [], []
            if target_idx_p is not None:
                points.append(
                    Point(x=float(self.trajectory.xys[target_idx_p, 0]), y=float(self.trajectory.xys[target_idx_p, 1]))
                )
                colors.append(ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
            if target_idx_v is not None:
                points.append(
                    Point(x=float(self.trajectory.xys[target_idx_v, 0]), y=float(self.trajectory.xys[target_idx_v, 1]))
                )
                colors.append(ColorRGBA(a=1.0))
            closest_point_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=0,
                points=points,
                colors=colors,
                scale=Vector3(x=0.06, y=0.06),
                ns="closest point",
            )
            marker_list.append(closest_point_marker)

        if self.state is not None:
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

    def transform_trajectory(
        self, trajectory: Trajectory, target_frame: str, source_frame: str
    ) -> Optional[Trajectory]:
        """Transforms a trajectory to another frame by looking a ROS transform."""
        if target_frame == source_frame:
            return trajectory

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

    def trajectory_callback(self, msg: TrajectoryMsg) -> None:
        """Callback for the trajectory topic."""
        self.last_target_idx = 0

        if not msg.path.poses:
            # This basically disables the trajectory follower
            self.trajectory = None
            return

        traj_frame = msg.path.header.frame_id
        positions = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.path.poses])
        quaternions = [p.pose.orientation for p in msg.path.poses]
        yaws = np.array(
            [euler_from_quaternion(*map(lambda attr: getattr(q, attr), ["x", "y", "z", "w"]))[2] for q in quaternions]
        )
        velocities = (
            np.array([v.linear.x for v in msg.velocities]) if len(msg.velocities) != 0 else np.zeros(len(positions))
        )
        accelerations = (
            np.array([a.linear.x for a in msg.accelerations])
            if len(msg.accelerations) != 0
            else np.zeros(len(positions))
        )
        if not (len(positions) == len(velocities) == len(accelerations)):
            self.get_logger().error("All lists in the message should contain the same number of entries")
            return
        stamps = np.array([pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9 for pose in msg.path.poses])
        trajectory = Trajectory(
            xys=positions,
            yaws=yaws,
            vs=velocities,
            accs=accelerations,
            stamps=stamps,
        )
        trajectory_transformed = self.transform_trajectory(trajectory, self.map_frame, traj_frame)
        if trajectory_transformed is not None:
            self.trajectory = trajectory_transformed

    def odom_callback(self, msg: Odometry) -> None:
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
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z
        odom_frame = msg.header.frame_id

        if self.map_frame == odom_frame:
            self.state = State(
                *position, yaw, linear_velocity, angular_velocity, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # type: ignore
            )
            return

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
            *transf_position, transf_yaw, linear_velocity, angular_velocity, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # type: ignore
        )

    def get_closest_idx(self) -> Tuple[int, int]:
        """Finds the index of the closest point on the trajectory"""
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys  # type: ignore
        dist = np.linalg.norm(dist_vec, axis=1)
        target_idx = int(np.argmin(dist))
        if self.last_target_idx > target_idx:
            target_idx = self.last_target_idx
        else:
            self.last_target_idx = target_idx

        return target_idx, target_idx

    def first_future_idx(self) -> Tuple[int, int]:
        """Finds the index of the first future goal state on the trajectory and the index of the state an x amount of
        seconds ahead of the current position."""
        current = self.get_clock().now().nanoseconds * 1e-9
        mask = self.trajectory.stamps > current  # type: ignore
        target_idx_v = np.argmax(mask) if mask.sum() > 0 else -1  # index of first True and else the last element

        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys[target_idx_v]
        par_vec = -np.array([np.cos(self.state.yaw), np.sin(self.state.yaw)])
        par_error = dist_vec @ par_vec
        goal_vel = self.trajectory.vs[target_idx_v] if self.trajectory.vs[target_idx_v] != 0.0 else 1e-6
        t_behind = min(par_error / goal_vel, 1.5)
        current_vel = self.state.v_lin if self.state.v_lin != 0.0 else 1e-6
        t_lag = min(self.goal_distance_lead / current_vel, 0.6)  # [s]
        mask = self.trajectory.stamps > (current - t_behind + t_lag)  # type: ignore
        target_idx_p = np.argmax(mask) if mask.sum() > 0 else -1  # index of first True and else the last element

        return int(target_idx_v), int(target_idx_p)

    def interpolate_target_velocity(self, target_idx: int) -> float:
        """Interpolate the target velocity between the waypoints."""
        if target_idx > 0:
            v_next, v_prev = self.trajectory.vs[target_idx], self.trajectory.vs[target_idx - 1]
            t_next, t_prev = self.trajectory.stamps[target_idx], self.trajectory.stamps[target_idx - 1]
            if not t_next == t_prev:
                t_cur = self.state.t
                target_vel = v_prev + (t_cur - t_prev) / (t_next - t_prev) * (v_next - v_prev)
            else:
                target_vel = self.trajectory.vs[target_idx]
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
        if self.state is None or self.trajectory is None:
            if self.state is None:
                self.get_logger().info(
                    "No state update received, so control is not applied", throttle_duration_sec=self.throttle_duration
                )
            if self.trajectory is None:
                self.get_logger().info(
                    "No trajectory available, so control is not applied", throttle_duration_sec=self.throttle_duration
                )
            self.visualize_trajectory()
            return

        # calculate target index
        if self.follow_mode == "position":
            target_idx_v, target_idx_p = self.get_closest_idx()
        elif self.follow_mode == "time":
            target_idx_v, target_idx_p = self.first_future_idx()

        if target_idx_v is None or target_idx_p is None:
            self.visualize_trajectory()
            return

        # Perpendicular distance error
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys[target_idx_p]
        perp_vec = -np.array([np.cos(self.state.yaw + np.pi / 2), np.sin(self.state.yaw + np.pi / 2)])
        perp_error = dist_vec @ perp_vec

        # Calculate the linear acceleration of the normalised velocity with a P-controller
        # Target velocity is directly tracked
        target_vel = self.interpolate_target_velocity(target_idx_v)
        if self.control_mode == "acceleration":  # use the trajectory velocity and linear acceleration
            acc_traject = self.trajectory.accs[target_idx_v] / self.vel_pwm_rate  # m/s2 to pwm/s
            acc_error = self.p_control_lin_velocity(target_vel, self.state.v_lin)  # Requires typically a lower p value
            acc_linear = acc_traject + acc_error
        elif self.control_mode == "velocity":  # only use the trajectory velocity
            acc_linear = self.p_control_lin_velocity(target_vel, self.state.v_lin)

        # Check if the normalized linear acceleration stays within the boundaries: [-max_acceleration, max_acceleration]
        if abs(acc_linear) > self.max_norm_acc:
            self.get_logger().warn(
                f"Maximum normalized linear acceleration exceeded: {acc_linear:.3f} > {self.max_norm_acc:.3f}"
            )
            acc_linear = np.clip(acc_linear, -self.max_norm_acc, self.max_norm_acc)

        # Check if the normalized velocity stays within the boundaries: [0, max_velocity]
        self.output_vel += acc_linear / self.control_frequency
        if abs(self.output_vel) > self.max_norm_vel:
            self.get_logger().warn(
                (
                    f"Maximum normalized velocity exceeded: {self.output_vel:.3f} > {self.max_norm_vel:.3f}"
                    + f"Speed most likely not reachable: normalized linear acceleration = {acc_linear}"
                ),
                throttle_duration_sec=self.throttle_duration,
            )
            self.output_vel = np.clip(self.output_vel, -self.max_norm_vel, self.max_norm_vel)
        elif self.output_vel < 0.0:
            self.get_logger().warn(
                f"Normalized velocity is negative; {self.output_vel:.3f} < {0.}",
                throttle_duration_sec=self.throttle_duration,
            )
            self.output_vel = 0.0

        # Calculate the steering angle with a Stanley controller: ccw positive
        delta = self.stanley_controller(self.trajectory.yaws[target_idx_p], perp_error)
        if abs(delta) > self.max_steering_angle:
            if self.state.v_lin > 0.01:  # low velocities result in high delta
                self.get_logger().warn(f"delta > max_steering_angle; abs({delta:.3f}) > {self.max_steering_angle:.3f}")
            delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)

        # convert delta to angular velocity goal
        self.target_angular_velocity = self.delta_to_angular_velocity(delta, self.state.v_lin)
        acc_v_delta = self.pd_control_ang_velocity(self.target_angular_velocity, self.state.v_ang)
        self.output_v_delta += acc_v_delta / self.control_frequency

        # convert the steering angle to a speed difference of the wheels: v_delta = (v / 2) * (W / L) * tan(delta)
        # v_delta is proportional to the velocity, so this all works with normalized velocities
        # v_delta = self.output_vel / 2 * self.wheel_base_W / self.wheel_base_L * np.tan(delta)

        if abs(self.output_v_delta) > (1 - self.max_norm_vel):
            self.get_logger().warn(
                f"Normalized velocity delta for steering is to large;"
                + f"{self.output_v_delta:.3f} < {(1 - self.max_norm_vel):.3f}",
            )
            self.output_v_delta = np.clip(self.output_v_delta, self.max_norm_vel - 1, 1 - self.max_norm_vel)

        v_left, v_right = self.output_vel - self.output_v_delta, self.output_vel + self.output_v_delta
        if abs(v_left) > 1 or abs(v_right) > 1:
            self.get_logger().error(f"SHOULD NOT HAPPEN Motor values > 1 due to steering: {v_left=}, {v_right=}")
            v_left, v_right = self.scale_motor_vels(v_left, v_right)

        # The normalized velocity [0, 1] can be directly applied on the motors
        v_left, v_right = int(255 * v_left), int(255 * v_right)
        # send 1 when the actual value is 0, since 0 results in freewheeling and 1 in braking
        v_left = v_left if v_left != 0 else 1
        v_right = v_right if v_right != 0 else 1
        motor_command = Int16MultiArray(data=[v_left, v_right, 0, 0])  # the hat_node expects an array of length 4
        self.motor_publisher.publish(motor_command)

        # debug_string = (
        #     f"v_target = {self.trajectory.vs[target_idx]:.3f}, v_current = {self.state.v_lin:.3f},"
        #     + f" v_norm = {self.output_vel:.3f},\n a_target = {self.trajectory.accs[target_idx]:.3f}"
        #     + f" acc_current = {acceleration}"
        # )
        # debug_string = f"goal_orient = {self.trajectory.yaws[target_idx]}, actual_orient = {self.state.yaw}"
        self.visualize_trajectory(target_idx_p=target_idx_p, target_idx_v=target_idx_v)

        if self.do_store_data:
            self.store_data_on_disk(target_idx_p, target_idx_v)

    def p_control_lin_velocity(self, target: float, current: float) -> float:
        """P controller for the linear velocity"""
        return self.lin_velocity_Kp * (target - current)

    def pd_control_ang_velocity(self, target: float, current: float) -> float:
        """PD controller for the angular velocity"""
        error = target - current
        previous_error = self.prev_ang_error if self.prev_ang_error is not None else error
        self.prev_ang_error = error
        return self.ang_velocity_Kp * error + self.ang_velocity_Kd * (error - previous_error) * self.control_frequency

    def stanley_controller(self, goal_yaw, perp_error) -> float:
        """Calculates the steering angle in radians according Stanley steering algorithm"""
        # theta_e corrects the heading error
        theta_e = angle_mod(goal_yaw - self.state.yaw)
        # theta_d corrects the cross track error; np.arctan2(y, x) -> so x and y are inverted (arctan(y / x))
        theta_d = np.arctan2(self.steering_k * perp_error, self.state.v_lin)

        return theta_e + theta_d  # [rad]

    def delta_to_angular_velocity(self, delta: float, linear_velocity: float) -> float:
        """Converts a steering angle to a desired angular velocity"""
        corner_radius = self.wheel_base_L / np.tan(delta)
        angular_velocity = linear_velocity / corner_radius

        return angular_velocity

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

    def store_data_on_disk(self, target_idx_p, target_idx_v):
        """Stores the data on disk to investigate the trajectory follower performance."""
        data_dict = {
            "current": self.state,
            "target": State(
                *self.trajectory.xys[target_idx_p],
                self.trajectory.yaws[target_idx_p],
                self.trajectory.vs[target_idx_v],
                self.target_angular_velocity,
                self.trajectory.stamps[target_idx_v],
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
