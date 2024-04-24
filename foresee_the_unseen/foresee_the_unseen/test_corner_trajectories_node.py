import rclpy
import os
import math
import pickle
import numpy as np
from rclpy.node import Node
from rclpy.time import Time as RosTime, Duration
from typing import List, Callable, Tuple, Type, Optional, Literal, TypeAlias
import nptyping as npt
from dataclasses import dataclass

from nav_msgs.msg import Odometry, Path
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, Twist, Vector3, TransformStamped
from std_msgs.msg import Header, Int16MultiArray

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import create_log_directory
from foresee_the_unseen.lib.planner import Planner

# Log directory
LOG_DIR = "/home/christiaan/thesis/test_trajectories_log_files"

# Test planner settings
TEST_DISTANCE = 5  # [m]
DX_WAYPOINTS = 0.1  # [m]
TURN_CORNER_RADIUS = 0.5  # [m]
DT = 0.25  # [s]
MAX_DIST_CORNER_SMOOTHING = 0.2  # [m]
FREQUENCY_TRAJECTORY = 20
MARGIN_VELOCITY = 0.001  # [m/s]
MARGIN_POSITION = 0.5  # [m/s]
MARGIN_ORIENTATION = 0.2  # [rad]
REFERENCE_SPEED = 0.3  # [m/s]
NO_OF_TRAJECTORIES = 2
MARGIN_TRAJECT_DISTANCE = 0.5  # [m]

# Velocity profiles settings
REF_VEL = 0.25
MAX_ACC = 0.25
MAX_DEC = 0.25
DT = 0.25
MAX_DIST_CORNER_SMOOTHING = 0.2

# Turn trajectory
STRAIGHT_PART_LENGTH = 2

# vels = [0.2, 0.3, 0.4, 0.5]
# radii = [0.5, 1.0, 1.5]
# long_errors = [0.05, 0.1, 0.15, 0.2, 0.25]

vels = [0.2, 0.4]
radii = [1.0, 1.5]
long_errors = [0.1, 0.2, 0.3, 0.4]

CONFIGS = [{"velocity": v, "radius": r, "longitudinal_error": e} for r in radii for v in vels for e in long_errors]

# static transfrom command line:
# ros2 run tf2_ros static_transform_publisher --x 2.0 --y -0.2 --frame-id map --child-frame-id test_traject
# ros2 run tf2_ros static_transform_publisher --x 0.3 --frame-id map --child-frame-id test_traject

# robot command
# ros2 launch foresee_the_unseen bringup_robot_launch.py use_ekf:=true follow_traject:=true slam_mode:=elsewhere
# ros2 launch foresee_the_unseen bringup_robot_launch.py use_ekf:=false follow_traject:=true slam_mode:=disabled

# laptop command
# ros2 launch foresee_the_unseen bringup_laptop_launch.py slam_mode_robot:=localization use_ekf_robot:=true map_file:=bed_room use_obstacles:=false use_foresee:=false
# ros2 launch foresee_the_unseen bringup_laptop_launch.py slam_mode_robot:=localization use_ekf_robot:=true map_file:=on_the_floor use_obstacles:=false use_foresee:=false


PlannerStates: TypeAlias = Literal["initialized", "towards corner", "following trajectory", "full turn"]


@dataclass
class State:
    position: npt.NDArray[npt.Shape["2"], npt.Float]  # [m]
    orientation: float  # [rad]
    velocity: float  # [m/s]
    time: Optional[RosTime] = None

    def pickable(self):
        return State(
            position=self.position,
            orientation=self.orientation,
            velocity=self.velocity,
            time=self.time.to_msg(),  # type: ignore
        )


@dataclass
class Trajectory:
    positions: npt.NDArray[npt.Shape["N, 2"], npt.Float]
    orientations: npt.NDArray[npt.Shape["N"], npt.Float]
    velocities: npt.NDArray[npt.Shape["N"], npt.Float]
    time_stamps: npt.NDArray[npt.Shape["N"], npt.Float]

    def transform(self, translation: np.ndarray = np.zeros(2), rotation: float = 0.0):
        r_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        t_matrix = np.zeros((3, 3))
        t_matrix[:2, :2] = r_matrix
        t_matrix[:2, 2] = translation
        t_matrix[2, 2] = 1

        positions_tf = (t_matrix @ np.vstack((self.positions.T, np.ones(len(self.positions)))))[:2].T
        orientations_tf = self.orientations + rotation

        return Trajectory(positions_tf, orientations_tf, self.velocities, self.time_stamps)


class TestStraightTrajectoriesNode(Node):
    """Plans trajectories in the `test_traject` frame."""

    def __init__(self):
        super().__init__("test_straight_trajectories_node")
        self.traject_frame = "test_traject"
        self.planner_state: PlannerStates = "initialized"
        # self.planner_state: PlannerStates = "full turn"
        self.base_log_dir = create_log_directory(LOG_DIR)

        # self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 5)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 5)
        self.publisher_trajectory = self.create_publisher(TrajectoryMsg, "trajectory", 5)
        self.publisher_motor_cmd = self.create_publisher(Int16MultiArray, "cmd_motor", 5)

        self._state: Optional[State] = None
        self._goal_state: Optional[State] = None
        self._trajectory_msg = None
        self.is_at_start = True  # true if traveling in the positive x direction, otherwise False

        self.config_idx = 0

        # Transformer listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.planner_timer = self.create_timer(1 / FREQUENCY_TRAJECTORY, self.run_state_machine)
        self.create_timer(1 / 5, self.republish_trajectory)

    @property
    def state(self) -> State:
        if self._state is None:
            raise AttributeError()
        return self._state

    @state.setter
    def state(self, state: State) -> None:
        self._state = state

    @property
    def goal_state(self) -> State:
        if self._goal_state is None:
            raise AttributeError()
        return self._goal_state

    @goal_state.setter
    def goal_state(self, goal_state: State) -> None:
        self._goal_state = goal_state

    @property
    def waypoints(self) -> npt.NDArray[npt.Shape["N, 2"], npt.Float]:
        return self._waypoints

    @waypoints.setter
    def waypoints(self, waypoints: npt.NDArray[npt.Shape["N, 2"], npt.Float]) -> None:
        self._waypoints = waypoints
        diff_points = np.diff(waypoints, axis=0)
        dist_bw_points = np.hypot(*diff_points.T)
        self.dist_along_points = np.hstack(([0], np.cumsum(dist_bw_points)))
        self.orient_bw_points = np.arctan2(*diff_points.T[::-1])

    @property
    def trajectory_msg(self) -> TrajectoryMsg:
        if self._trajectory_msg is None:
            raise AttributeError
        return self._trajectory_msg

    @trajectory_msg.setter
    def trajectory_msg(self, trajectory_msg: TrajectoryMsg) -> None:
        self._trajectory_msg = trajectory_msg

    def plan_test_trajectories(
        self, waypoints: npt.NDArray[npt.Shape["N, 2"], npt.Float], no_of_trajectories: int
    ) -> List[Trajectory]:
        dist_to_goal = np.hypot(*np.diff(waypoints, axis=0).T).sum() - MARGIN_TRAJECT_DISTANCE
        velocity_profiles = Planner._generate_scalar_velocity_profiles(
            velocity=0,
            max_dec=MAX_DEC,
            max_acc=MAX_ACC,
            dt=DT,
            time_horizon=100,
            reference_speed=REF_VEL,
            number_of_trajectories=no_of_trajectories,
            dist_to_goal=dist_to_goal,
        )
        # TODO: remove states at the end with 0 velocity, except for one
        velocity_profiles_rm_zeros = []
        for v_profile in velocity_profiles[:-1]:  # remove all zeros profile
            idx_first_non_zero = np.argmax(v_profile[::-1] > 0)
            if idx_first_non_zero > 1:
                v_profile = v_profile[: -(idx_first_non_zero - 1)]
            velocity_profiles_rm_zeros.append(v_profile)

        trajectories = [
            self.velocity_profile_to_trajectory(v_profile, self.waypoints) for v_profile in velocity_profiles_rm_zeros
        ]

        self.trajectory_to_test_idx = 0

        ############### PLOT the trajectories #############
        # import matplotlib.pyplot as plt

        # grid_size = int(np.sqrt(no_of_trajectories)) + 1
        # fig, axs = plt.subplots(grid_size, grid_size)
        # for trajectory, ax in zip(trajectories, axs.flatten()):
        #     # trajectory = trajectory.transform(translation=np.array([1, 2]), rotation=np.pi *2 / 3)
        #     ax.scatter(*trajectory.positions.T)
        #     for p, yaw in zip(trajectory.positions, trajectory.orientations):
        #         unit_vec = np.array([np.cos(yaw), np.sin(yaw)])
        #         vec_length = 0.2
        #         ax.plot(*np.vstack((p - unit_vec * vec_length, p + unit_vec * vec_length)).T, 'k')

        # plt.show()
        ########################## END ######################

        return trajectories

    def republish_trajectory(self) -> None:
        try:
            self.publisher_trajectory.publish(self.trajectory_msg)
        except AttributeError:
            return

    def odom_callback(self, msg: Odometry) -> None:
        """The odometry callback"""
        odom_frame = msg.header.frame_id
        try:
            t_odom_traject = self.tf_buffer.lookup_transform(self.traject_frame, odom_frame, RosTime())
        except TransformException as e:
            self.get_logger().info(f"Could not transform {odom_frame} to {self.traject_frame}: {e}")
            return

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
        yaw = self.yaw_from_quaternion(*quaternion)
        velocity = msg.twist.twist.linear.x
        state = State(
            position=np.array(position),
            orientation=yaw,
            velocity=velocity,
            time=RosTime(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec),
        )
        self.state = self.transform_state(state, t_odom_traject)

    def publish_trajectory(self, trajectory: Optional[Trajectory]) -> None:
        """Publish a trajectory"""
        if trajectory is None:
            msg = TrajectoryMsg()
        else:
            msg = self.create_trajectory_msg(trajectory)
            time_goal = trajectory.time_stamps[-1]
            self.goal_state = State(
                position=trajectory.positions[-1],
                orientation=trajectory.orientations[-1],
                velocity=trajectory.velocities[-1],
                time=self.get_clock().now() + Duration(seconds=int(time_goal), nanoseconds=int((time_goal % 1) * 1e9)),
            )
        self.publisher_trajectory.publish(msg)
        self.trajectory_msg = msg
        # self.get_logger().info("Trajectory published")

    @staticmethod
    def make_angles_continuous(angles):
        """Remove angles switching from positive to negative around pi and -pi"""
        correct_angle = np.zeros_like(angles)
        angle_diff = np.insert(np.diff(angles), 0, 0)
        correct_angle[angle_diff < -np.pi] = 2 * np.pi
        correct_angle[angle_diff > np.pi] = -2 * np.pi
        correct_angle = np.cumsum(correct_angle)
        return angles + correct_angle

    @staticmethod
    def normalize_angles(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def quaternion_from_yaw(yaw):
        return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    @staticmethod
    def yaw_from_quaternion(x, y, z, w):
        t1 = +2.0 * (w * z + x * y)
        t2 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t1, t2)

    @staticmethod
    def transform_positions_orientations(
        positions: np.ndarray, orientations: np.ndarray, translation: np.ndarray = np.zeros(2), rotation: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        r_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        t_matrix = np.zeros((3, 3))
        t_matrix[:2, :2] = r_matrix
        t_matrix[:2, 2] = translation
        t_matrix[2, 2] = 1

        return (t_matrix @ np.vstack((positions.T, np.ones(len(positions)))))[:2].T, orientations + rotation

    @staticmethod
    def transform_state(state: State, transform: TransformStamped) -> State:
        quaternion = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        yaw = TestStraightTrajectoriesNode.yaw_from_quaternion(*quaternion)
        translation = [
            transform.transform.translation.x,
            transform.transform.translation.y,
        ]
        r_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        t_matrix = np.zeros((3, 3))
        t_matrix[:2, :2] = r_matrix
        t_matrix[:2, 2] = translation
        t_matrix[2, 2] = 1

        return State(
            position=(t_matrix @ np.append(state.position, 1))[:2],
            orientation=state.orientation + yaw,
            velocity=state.velocity,
            time=state.time,
        )

    @staticmethod
    def velocity_profile_to_trajectory(
        velocity_profile: np.ndarray,
        waypoints: np.ndarray,  # first waypoint is the current position of the robot
        dt: float = DT,
        max_dist_corner_smoothing: float = MAX_DIST_CORNER_SMOOTHING,
    ) -> Trajectory:
        assert velocity_profile.ndim == 1

        diff_points = np.diff(waypoints, axis=0)
        dist_bw_points = np.hypot(*diff_points.T)
        dist_along_points = np.hstack(([0], np.cumsum(dist_bw_points)))
        orient_bw_points = np.arctan2(*diff_points.T[::-1])
        orient_bw_points = TestStraightTrajectoriesNode.make_angles_continuous(orient_bw_points)

        dist = np.cumsum(velocity_profile * dt)
        x_coords = np.interp(dist, dist_along_points, waypoints[:, 0])
        y_coords = np.interp(dist, dist_along_points, waypoints[:, 1])
        positions = np.vstack((x_coords, y_coords)).T

        # The trajectory orientation on the parts between the waypoints is constant, but the trajectory  orientation
        #  around the waypoints is smoothed. The smoothening happens within a distance of the minimum of
        #  (`self.max_dist_corner_smoothing`) and half of the length of the part between two waypoints.
        dist_along_bef_points = dist_along_points[1:-1] - np.minimum(
            np.diff(dist_along_points[:-1]) / 2,
            max_dist_corner_smoothing,
        )
        dist_along_aft_points = dist_along_points[1:-1] + np.minimum(
            np.diff(dist_along_points[1:]) / 2,
            max_dist_corner_smoothing,
        )
        dist_along_bef_aft_points = np.vstack((dist_along_bef_points, dist_along_aft_points)).T.flatten()
        orientations_at_points = np.repeat(orient_bw_points, 2)[1:-1]
        orientations = np.interp(dist, dist_along_bef_aft_points, orientations_at_points)
        orientations = TestStraightTrajectoriesNode.normalize_angles(orientations)
        time_stamps = (np.arange(len(velocity_profile)) + 1) * dt

        return Trajectory(
            positions=positions,
            orientations=orientations,
            velocities=velocity_profile,
            time_stamps=time_stamps,
        )

    @staticmethod
    def get_reposition_trajectory(
        current_position: npt.NDArray[npt.Shape["2"], npt.Float],
        translation: npt.NDArray[npt.Shape["2"], npt.Float] = np.zeros(2),
        rotation: float = 0.0,
        lin_distance: float = 0.3,
        min_corner_radius: float = TURN_CORNER_RADIUS,
        dx_waypoints: float = 0.05,
        reference_speed: float = REFERENCE_SPEED,
        max_acceleration: float = 0.1,
        dt: float = 0.25,
    ) -> Trajectory:
        """Generate a trajectory to reposition the robot"""
        start_angle, end_angle = np.pi / 2, np.pi / 6
        angles = np.linspace(
            start_angle, end_angle, int(abs(start_angle - end_angle) * min_corner_radius / dx_waypoints) + 2
        )
        pts_corner_start = min_corner_radius * np.array([np.cos(angles), np.sin(angles)]).T + [0, -min_corner_radius]

        start_angle, end_angle = -np.pi * 5 / 6, np.pi * 5 / 6
        angles = np.linspace(
            start_angle, end_angle, int(abs(start_angle - end_angle) * min_corner_radius / dx_waypoints) + 2
        )
        pts_corner_circle = min_corner_radius * np.array([np.cos(angles), np.sin(angles)]).T + [
            np.sqrt(3) * min_corner_radius,
            0,
        ]

        start_angle, end_angle = -np.pi / 6, -np.pi / 2
        angles = np.linspace(
            start_angle, end_angle, int(abs(start_angle - end_angle) * min_corner_radius / dx_waypoints) + 2
        )
        pts_corner_end = min_corner_radius * np.array([np.cos(angles), np.sin(angles)]).T + [0, min_corner_radius]

        pts_turn = np.vstack((pts_corner_start[:-1], pts_corner_circle[:-1], pts_corner_end))

        pts_lin_bef = np.linspace([0, 0], [lin_distance, 0], round(lin_distance / dx_waypoints) + 1)
        pts_lin_aft = pts_lin_bef[::-1]
        pts_total = np.vstack((pts_lin_bef[:-1], pts_turn + [lin_distance, 0], pts_lin_aft[1:]))

        pts_total, _ = TestStraightTrajectoriesNode.transform_positions_orientations(
            pts_total, np.array([0]), translation, rotation
        )
        pts_total = np.insert(pts_total, 0, current_position, axis=0)

        total_distance = np.linalg.norm(np.diff(pts_total, axis=0), axis=1).sum()
        # T = v/a + x/v
        total_time = reference_speed / max_acceleration + total_distance / reference_speed
        total_time_steps = int(total_time / dt) + 1
        max_acc_vel_profile = max_acceleration * np.arange(total_time_steps) * dt
        velocity_profile = np.minimum(
            max_acc_vel_profile[::-1], np.minimum(np.full(total_time_steps, reference_speed), max_acc_vel_profile)
        )

        trajectory = TestStraightTrajectoriesNode.velocity_profile_to_trajectory(
            velocity_profile=velocity_profile,
            waypoints=pts_total,
        )

        return trajectory

    def save_state_to_disk(self, directory: str):
        """Store the state to disk"""
        save_time = self.get_clock().now if self.state.time is None else self.state.time
        time_stamp = str(save_time.seconds_nanoseconds()[0]) + "_" + str(save_time.seconds_nanoseconds()[1])
        filename = os.path.join(directory, f"step {time_stamp}.pickle")
        with open(filename, "wb") as handle:
            pickle.dump(self.state.pickable(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_trajectory_msg(self, trajectory: Trajectory) -> TrajectoryMsg:
        start_stamp = self.get_clock().now()
        header_path = Header(stamp=start_stamp.to_msg(), frame_id=self.traject_frame)
        pose_stamped_list = []
        twist_list = []
        for position, orientation, velocity, time_stamp in zip(
            trajectory.positions, trajectory.orientations, trajectory.velocities, trajectory.time_stamps
        ):
            quaternion = Quaternion(
                **{k: v for k, v in zip(["x", "y", "z", "w"], self.quaternion_from_yaw(orientation))}
            )
            position = Point(x=float(position[0]), y=float(position[1]))
            time_diff = time_stamp
            header_pose = Header(
                stamp=(start_stamp + Duration(nanoseconds=int(time_diff * 1e9))).to_msg()
            )
            pose_stamped_list.append(
                PoseStamped(header=header_pose, pose=Pose(position=position, orientation=quaternion))
            )
            twist_list.append(Twist(linear=Vector3(x=float(velocity))))

        return TrajectoryMsg(path=Path(header=header_path, poses=pose_stamped_list), velocities=twist_list)

    def return_to_boundary(self, do_return_to_start: bool) -> None:
        """Return to either the first (do_return_to_start==True) or last waypoint with the right orientation."""
        idx_waypoint = 0 if do_return_to_start else -1

        pt_current, pt_goal = self.state.position, self.waypoints[idx_waypoint]
        positions = np.array([pt_current, (pt_current + pt_goal) / 2, pt_goal])
        reference_speed: float = REFERENCE_SPEED
        max_acceleration: float = 0.1
        dt: float = 0.25

        total_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        # T = v/a + x/v
        total_time = reference_speed / max_acceleration + total_distance / reference_speed
        total_time_steps = int(total_time / dt) + 1
        max_acc_vel_profile = max_acceleration * np.arange(total_time_steps) * dt
        velocity_profile = np.minimum(
            max_acc_vel_profile[::-1], np.minimum(np.full(total_time_steps, reference_speed), max_acc_vel_profile)
        )

        trajectory = TestStraightTrajectoriesNode.velocity_profile_to_trajectory(
            velocity_profile=velocity_profile,
            waypoints=positions,
        )

        self.publish_trajectory(trajectory)

    def has_goal_position(
        self, margin_position: float = MARGIN_POSITION, margin_orientation: float = MARGIN_ORIENTATION
    ) -> bool:
        return bool(
            np.linalg.norm(self.state.position - self.goal_state.position) < margin_position
            and np.abs(self.normalize_angles(self.state.orientation - self.goal_state.orientation)) < margin_orientation
        )

    def has_goal_velocity(self, margin_velocity: float = MARGIN_VELOCITY) -> bool:
        return bool(np.abs(self.state.velocity - self.goal_state.velocity) < margin_velocity)

    def has_passed_goal(self) -> bool:
        # rotate the gaol orientatin by 90 degrees and check in which halfspace
        line_orientation = self.goal_state.orientation + np.pi / 2
        goal_line = np.array([np.cos(line_orientation), np.sin(line_orientation)])
        return bool(np.cross(self.state.position - self.goal_state.position, goal_line) > 0)

    @staticmethod
    def get_corner(straight_part: float, radius: float, dx_waypoints: float = DX_WAYPOINTS) -> np.ndarray:
        thetas = np.linspace(0, np.pi / 2, int(np.pi / 2 * radius / dx_waypoints) + 1)
        pts_corner = np.array([np.cos(thetas - np.pi / 2), np.sin(thetas - np.pi / 2)]).T * radius + np.array(
            [straight_part, radius]
        )
        pts_corner = np.array([np.cos(thetas - np.pi / 2), np.sin(thetas - np.pi / 2)]).T * radius + np.array(
            [straight_part, radius]
        )

        return np.vstack(([0, 0], pts_corner, [straight_part + radius, straight_part + radius]))

    def update_waypoints(self) -> None:
        self.publish_trajectory(None)
        if self.config_idx == len(CONFIGS):
            exit(0)

        self.config = CONFIGS[self.config_idx]
        self.velocity, self.radius, self.long_error = self.config.values()
        self.config_idx += 1
        waypoints = self.get_corner(straight_part=STRAIGHT_PART_LENGTH, radius=self.radius)
        self.waypoints = waypoints[:: 1 if self.is_at_start else -1]

        # make a new log directory for the states and save the waypoints and configuration also to disk
        self.trajectory_log_dir = os.path.join(self.base_log_dir, f"trajectory {self.config_idx}")
        os.mkdir(self.trajectory_log_dir)
        with open(os.path.join(self.trajectory_log_dir, f"waypoints.pickle"), "wb") as handle:
            waypoints = self.waypoints
            pickle.dump(waypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.trajectory_log_dir, f"config.pickle"), "wb") as handle:
            waypoints = self.waypoints
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_corner_trajectory_with_longitudinal_error(self) -> None:
        """Calculates and publishes the corner trajectory."""
        # publish a trajectory with x amount of longitudinal error
        t_error = self.long_error / self.velocity
        # make a velocity_profile which starts at the reference speed, but slows down at the end
        dist_to_goal = np.hypot(*np.diff(self.waypoints[1:], axis=0).T).sum()
        total_time = dist_to_goal / self.velocity + self.velocity / (2 * MAX_DEC)
        constant_velocity = np.full(int(total_time / DT), self.velocity)
        ramp_down = np.arange(len(constant_velocity)) * MAX_DEC * DT
        velocity_profile = np.minimum(constant_velocity, ramp_down[::-1])
        trajectory = self.velocity_profile_to_trajectory(velocity_profile, self.waypoints[1:])
        ramp_idcs = math.ceil(self.velocity / MAX_DEC / DT)  # ramp_time / dt
        trajectory.time_stamps[:-ramp_idcs] -= t_error
        # shift time stamps according to t_error
        self.publish_trajectory(trajectory)

        with open(os.path.join(self.trajectory_log_dir, f"trajectory.pickle"), "wb") as handle:
            pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_trajectory_towards_corner(self) -> None:
        """Calculates and publishes the trajectory towards the corner."""
        self.update_waypoints()
        position = self.waypoints[1]
        orientation = np.arctan2(*(self.waypoints[1] - self.waypoints[0])[::-1])
        trajectory = Trajectory(
            positions=np.array([position]),
            orientations=np.array([orientation]),
            velocities=np.array([self.velocity]),
            time_stamps=np.array([10000]),
        )
        self.publish_trajectory(trajectory)

    def publish_straight_line(self) -> None:
        waypoints = np.array([[0, 0], [10, 0]])
        velocity_profile = np.full(500, 0.1)
        trajectory = self.velocity_profile_to_trajectory(velocity_profile, waypoints)
        self.publish_trajectory(trajectory)

    def run_state_machine(self) -> None:
        """The function that controls the state machine."""
        try:
            self.state
        except AttributeError:
            self.get_logger().info("No state available")
            return

        prev_state = self.planner_state
        if self.planner_state == "initialized":
            self.planner_state = "towards corner"
            self.set_trajectory_towards_corner()
        elif self.planner_state == "full turn":
            cmd_limit = 25
            kp = 0.1
            orient_error = self.normalize_angles(self.goal_state.orientation - self.state.orientation)

            if abs(orient_error) < 0.1:
                motor_cmd = 0
                self.get_logger().info("Full turn made")
                self.set_trajectory_towards_corner()
                self.planner_state = "towards corner"
            elif orient_error > 0:
                motor_cmd = np.clip(kp * orient_error * 255, cmd_limit, 255)
            else:
                motor_cmd = np.clip(kp * orient_error * 255, -255, -cmd_limit)

            self.publisher_motor_cmd.publish(Int16MultiArray(data=[int(-motor_cmd), int(motor_cmd), 0, 0]))
        elif self.planner_state == "towards corner":
            if self.has_passed_goal():
                self.get_logger().info("corner reached")
                self.set_corner_trajectory_with_longitudinal_error()
                self.planner_state = "following trajectory"
        elif self.planner_state == "following trajectory":
            self.save_state_to_disk(self.trajectory_log_dir)
            # if self.has_goal_position() and self.has_goal_velocity():
            if self.has_passed_goal():
                self.get_logger().info("Trajectory completed")
                self.publish_trajectory(None)
                self.goal_state.orientation += np.pi
                self.is_at_start = not self.is_at_start
                self.planner_state = "full turn"
        else:
            raise NameError

        if self.planner_state != prev_state:
            self.get_logger().info(f"State changed: `{prev_state}` ==> `{self.planner_state}`")


def main(args=None):
    rclpy.init(args=args)

    test_straight_trajectories_node = TestStraightTrajectoriesNode()

    rclpy.spin(test_straight_trajectories_node)
    test_straight_trajectories_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
