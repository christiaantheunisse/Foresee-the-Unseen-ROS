from rclpy.node import Node
import rclpy
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
from std_msgs.msg import Int16MultiArray, ColorRGBA, Header
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

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value
        self.odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value

        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.traj_marker_topic = self.get_parameter("trajectory_marker_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.motor_cmd_topic = self.get_parameter("motor_command_topic").get_parameter_value().string_value

        self.do_visual_traj = self.get_parameter("do_visualize_trajectory").get_parameter_value().bool_value

        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        self.wheel_base_W = self.get_parameter("wheel_base_width").get_parameter_value().double_value
        self.wheel_base_L = self.get_parameter("wheel_base_length").get_parameter_value().double_value

        # self.velocity_PID = self.get_parameter("velocity_PID").get_parameter_value().double_array_value
        self.velocity_Kp = self.get_parameter("velocity_p").get_parameter_value().double_value
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

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)
        # self.create_subscription(TrajectoryMsg, self.trajectory_topic, self.trajectory_callback, 5)
        self.create_timer(0.5, self.trajectory_callback)
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


    def visualize_trajectory(self, target_idx: Optional[int] = None) -> None:
        if not self.do_visual_traj:
            return

        marker_list = []
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)

        if self.trajectory is not None:
            point_list = []
            color_rgba_list = []
            cmap = mpl.cm.get_cmap("cool")
            for (x, y), v in zip(self.trajectory.xys, self.trajectory.vs):
                point_list.append(Point(x=x, y=y))
                r, g, b, a = cmap(len(point_list) / len(self.trajectory.xys))
                color_rgba_list.append(ColorRGBA(r=r, g=g, b=b, a=a))
            trajectory_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=8954,
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
                id=8955,
                points=[Point(x=self.trajectory.xys[target_idx, 0], y=self.trajectory.xys[target_idx, 1])],
                colors=[ColorRGBA(a=1.0)],
                scale=Vector3(x=0.06, y=0.06),
                ns="closest point",
            )
            marker_list.append(closest_point_marker)

        if self.state.x is not None:
            closest_point_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=8955,
                points=[Point(x=float(self.state.x), y=float(self.state.y))],
                colors=[ColorRGBA(a=1.0)],
                scale=Vector3(x=0.1, y=0.1),
                ns="state",
            )
            marker_list.append(closest_point_marker)

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

        positions = [[0.025, 0.0013], [0.0749, 0.0039], [0.1493, 0.0126], [0.2472, 0.0324], [0.366, 0.0708], [0.4792, 0.1234], [0.5852, 0.1893], [0.6814, 0.2691], [0.7668, 0.3601], [0.8406, 0.4608], [0.9016, 0.5697], [0.9489, 0.6853], [0.9804, 0.8062], [0.9963, 0.93], [1.0, 1.0549], [1.0, 1.1549], [1.0, 1.2299], [1.0, 1.2799], [1.0, 1.3049], [1.0, 1.3049]]
        orientations = [0.0524, 0.0524, 0.0977, 0.1978, 0.3228, 0.4479, 0.5729, 0.698, 0.823, 0.9481, 1.0732, 1.1982, 1.3233, 1.4483, 1.5213, 1.5265, 1.5305, 1.5331, 1.5344, 1.5344]
        velocities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        self.trajectory = Trajectory(positions, orientations, velocities)

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
        self.state = State(*transf_position, transf_yaw, velocity)

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
        dist_vec = np.array([self.state.x, self.state.y]) - self.trajectory.xys
        dist = np.linalg.norm(dist_vec, axis=1)
        target_idx = np.argmin(dist)
        if self.last_target_idx > target_idx:
            target_idx = self.last_target_idx
        else:
            self.last_target_idx = target_idx

        # perpendicular distance error
        perp_vec = -np.array([np.cos(self.state.yaw + np.pi / 2), np.sin(self.state.yaw + np.pi / 2)])
        perp_error = dist_vec[target_idx] @ perp_vec

        # calculate the acceleration
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

        # calculate the steering angle: ccw positive
        delta = self.stanley_controller(self.trajectory.yaws[target_idx], perp_error)
        if abs(delta) > self.max_steering_angle:
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
        motor_command = Int16MultiArray(data=[v_left, v_right, 0, 0])  # 4 motors are implemented by the hat_node
        self.motor_publisher.publish(motor_command)

        self.visualize_trajectory(target_idx)

    def p_control(self, target, current) -> float:
        """PD controller for the speed."""
        return self.velocity_Kp * (target - current)
        Kp, Ki, Kd = self.velocity_PID
        error = target - current
        derror = (error - self.prev_error) * self.control_frequency if self.prev_error is not None else 0
        self.prev_error = error

        return Kp * error + Kd * derror

    def stanley_controller(self, goal_yaw, perp_error) -> float:
        """Calculates the steering angle in radians according Stanley steering algorithm"""
        # theta_e corrects the heading error
        theta_e = angle_mod(goal_yaw - self.state.yaw)
        # theta_d corrects the cross track error
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
