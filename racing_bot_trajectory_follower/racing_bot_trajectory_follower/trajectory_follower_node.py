from rclpy.node import Node
import rclpy
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional

from nav_msgs.msg import Odometry
from std_msgs.msg import Int16MultiArray


@dataclass
class State:
    x: Optional[float] = None  # position along x-axis [m]
    y: Optional[float] = None  # position along y-axis [m]
    yaw: Optional[float] = None  # heading (0 along x-axis, ccw positive) [rad]
    v: Optional[float] = None  # velocity [m/s]


@dataclass
class Trajectory:
    xys: np.ndarray  # xy-positions [m] [[x, y], ...]
    yaws: np.ndarray  # headings [rad] [yaw, ...]
    vs: np.ndarray  # velocties [m/s] [v, ...]


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


class TrajectoryFollowerNode(Node):
    """
    This node implements Stanley steering control and PID speed control which is based on the code from the github
    page PythonRobotics:
        https://github.com/AtsushiSakai/PythonRobotics/blob/bd253e81060c6a11a944016706bd1d87ef72bded/PathTracking/stanley_controller/stanley_controller.py

    A futher explanation of the algorithm is given in the method: `apply_control()`
    """

    def __init__(self):
        # super().__init__("trajectory_follower_node")

        # parameters
        self.trajectory_topic = None
        self.odom_topic = "odom"
        self.motor_cmd_topic = "cmd_motor"
        self.control_frequency = 10  # [Hz]
        self.W = 0.145  # distance between the wheels on the rear axle [m]
        self.L = 0.1  # position of the front axle measured from the rear axle [m]

        self.velocity_Kp = 5.0  # velocity proportional gain
        self.steering_k = 0.5  # steering control gain

        # Temporary
        no_points = 100
        xys = np.linspace([0, 0], [5, 0], no_points)
        diffs = xys[1:] - xys[:-1]
        diffs = np.append(diffs, diffs[-1:], axis=0)
        yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
        vs = np.full(no_points, 0.4)

        self.trajectory = Trajectory(xys, yaws, vs)
        self.state = State()
        self.last_target_idx = 0
        self.output_vel = 0

        # self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)
        # self.motor_publisher = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, 10)
        # self.create_timer(1 / self.control_frequency, self.apply_control)

    def trajectory_callback(self, msg):
        self.last_target_idx = 0

    def odom_callback(self, msg: Odometry):
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

        self.state = State(*position, yaw, velocity)

    def apply_control(self):
        """
        Calculates the motor commands based on the trajectory and the position using a Stanley controller for the
        steering angle and a pid controller for the speed.

        The pid controller is tuned to output the acceleration for a normalized motor speed in the range [-1, 1].

        The Stanley controller is developed for Ackermann steering vehicles and not for a differential drive mobile
        robot (DDMR). The position of the front axle is chosen to be around the castor wheel, but this is a arbitrary
        decision. The algorithms tries to position the front axle on the path and the (fictional) front wheels with the
        path. The center of the baselink frame (and thus the position the odometry pose is pointing to) is the center
        of the rear axle.

        """
        if self.state.x is None or self.state.y is None or self.state.yaw is None or self.state.v is None:
            self.get_logger().warn("No state update received")

        # calculate target index
        dist = np.linalg.norm(np.array([self.state.x, self.state.y]) - self.trajectory.xys, axis=1)
        target_idx = np.argmin(dist)
        if self.last_target_idx > target_idx:
            target_idx = self.last_target_idx
        else:
            self.last_target_idx = target_idx

        # perpendicular distance error
        perp_vec = -np.array([np.cos(self.state.yaw + np.pi / 2), np.sin(self.state.yaw + np.pi / 2)])
        perp_error = dist[target_idx] @ perp_vec

        # calculate the acceleration
        acceleration = self.p_control(self.trajectory.vs[target_idx], self.state.v)
        self.output_vel += acceleration / self.control_frequency
        if abs(self.output_vel) > 1:
            self.output_vel = np.clip(self.output_vel, -1, 1)
            self.get_logger().warn(
                f"Acceleration is to high. Speed most likely not reachable: acceleration = {acceleration}"
            )

        # calculate the steering angle: ccw positive
        delta = self.stanley_controller(self.trajectory.yaws[target_idx], perp_error)
        # convert the steering angle to a speed difference of the wheels: v_delta = (v / 2) * (W / L) * tan(delta)
        v_delta = self.output_vel / 2 * self.W / self.L * np.tan(delta)

        # directly set the motor speed
        v_left, v_right = self.output_vel - v_delta, self.output_vel + v_delta
        if abs(v_left) > 1 or abs(v_right) > 1:
            v_left, v_right = self.scale_motor_vels(v_left, v_right)
            self.get_logger().warn(
                f"Maximum motor values exceed due to steering: [v_left, v_right] = {[v_left, v_right]}"
            )

        v_left, v_right = int(255 * v_left), int(255 * v_right)
        motor_command = Int16MultiArray(data=[v_left, v_right, 0, 0])  # 4 motors are implemented by the hat_node
        self.motor_publisher(motor_command)

        self.get_logger().info(f"Control: [v_left, v_right] = {v_left, v_right}")

    def p_control(self, target, current) -> float:
        """Proportional control for the speed."""
        return self.velocity_Kp * (target - current)

    def stanley_controller(self, goal_yaw, perp_error) -> float:
        """Calculates the steering angle"""
        # theta_e corrects the heading error
        theta_e = angle_mod(goal_yaw - self.state.yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.steering_k * perp_error, self.state.v)

        return theta_e + theta_d


def main(args=None):
    rclpy.init(args=args)

    trajectory_follower_node = TrajectoryFollowerNode()

    rclpy.spin(trajectory_follower_node)
    trajectory_follower_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
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
