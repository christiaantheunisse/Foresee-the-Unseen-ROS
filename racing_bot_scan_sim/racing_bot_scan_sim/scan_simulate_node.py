from __future__ import annotations
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from typing import Callable, Optional, Generator
import numpy as np
from collections import namedtuple
from scipy.spatial.transform import Rotation, Slerp
from math import inf

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Vector3, Point as PointMsg
from nav_msgs.msg import Odometry
from racing_bot_interfaces.msg import Trajectory
from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import Marker

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_sensor_msgs import transform_points
import tf2_geometry_msgs  # necessary to enable Buffer.transform
from visualization_msgs.msg import MarkerArray

Point = namedtuple("Point", "x y")


def halfspace_from_points(p1: np.ndarray, p2: np.ndarray):
    assert p1.shape == p2.shape == (2,), "Points should be 2 dimensional numpy arrays"
    a = Point(*p1)
    b = Point(*p2)

    # RHS of the line: x_coef * X + y_coef * Y <= b
    if a.x - b.x == 0:
        x_coef = a.y - b.y
        y_coef = 0
        const = -(a.x * b.y - b.x * a.y)
    else:
        x_coef = -(a.y - b.y) / (a.x - b.x)
        y_coef = 1
        const = (a.x * b.y - b.x * a.y) / (a.x - b.x)
        if a.x > b.x:
            x_coef, y_coef, const = -x_coef, -y_coef, -const

    return x_coef, y_coef, const


def matrices_from_cw_cvx_polygon(polygon):
    polygon = np.array(polygon)
    A, B = [], []
    for p1, p2 in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b, c = halfspace_from_points(p1, p2)
        A.append([a, b])
        B.append(c)

    return np.array(A), np.array(B)


def points_to_linesegments(points: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    for p1, p2 in zip(points, np.roll(points, -1, axis=0)):
        yield p1, p2


def ray_intersection_fraction(ray_endpoint: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates the fraction of the length along the ray where the intersections is, if any.

    Arguments:
        ray_endpoint -- the endpoint of the ray starting in the origin
        p1 -- one point of the linesegment making up the obstacle edge
        p2 -- the other point of the linesegment making up the obstacle edge

    Returns:
        None if no intersection and else the fraction [0, 1] of the length of the ray where the intersection is
    """
    assert p1.size == p2.size == ray_endpoint.size == 2 and p1.ndim == p2.ndim == ray_endpoint.ndim == 1
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [0, 0], ray_endpoint, p1, p2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    sign = np.sign(denominator)
    t_numerator = sign * ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))
    u_numerator = -sign * ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    denominator *= sign
    return t_numerator / denominator if 0 <= t_numerator <= denominator and 0 <= u_numerator <= denominator else inf


def find_intersec_ranges(angles: np.ndarray, obstacle_points: np.ndarray, max_range: float) -> np.ndarray:
    """Finds the angles and ranges for the rays that intersection with the obstacle

    Arguments:
        angles -- angles at which rays are send
        obstacle_points -- corner points of the obstacle between which edges are drawn.
        max_range -- maximum ray length

    Returns:
        the ranges which are either infinite or the intersection distance
    """
    # determine angle range to check
    # obstacle_angles = np.array([np.arctan2(*np.flip(p)) for p in obstacle_points])

    # mask_angles = (angles >= obstacle_angles.min()) & (angles <= obstacle_angles.max())
    # angles_to_obstacle = angles[mask_angles]
    # unit_vecs = np.vstack((np.cos(angles_to_obstacle), np.sin(angles_to_obstacle))).T
    unit_vecs = np.vstack((np.cos(angles), np.sin(angles))).T
    rays = unit_vecs * max_range

    intersection_distances = []
    for ray in rays:
        intersection_distances.append(
            np.min([ray_intersection_fraction(ray, A, B) for A, B in points_to_linesegments(obstacle_points)])
        )
    return np.array(intersection_distances) * max_range


class ScanSimulateNode(Node):
    """This nodes change a LaserScan message obtained from e.g. a Lidar. Measurements can be removed from the Lidar and
    moving obstacles can be added. Data for the moving obstacles are taken from topics with a Trajectory message."""

    def __init__(self):
        super().__init__("scan_simulate_node")

        self.declare_parameter("obstacle_frame", "planner")
        self.declare_parameter("scan_in_topic", "scan")
        self.declare_parameter(
            "namespaces",
            ["obstacle_car1", "obstacle_car2"],
            ParameterDescriptor(
                description="List with namespaces: [namespace1, namespace2, ...]. Trajectory topic is "
                "`namespace/trajectory` and Odometry topic is `namespace/odometry/filtered`"
            ),
        )
        self.declare_parameter("scan_out_topic", "scan/simulated")

        self.declare_parameter("do_simulate_obstacles", True)
        self.declare_parameter("do_visualize", True)
        self.declare_parameter("do_filter_scan", True)
        self.declare_parameter(
            "environment_boundary",
            # [-2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0],
            [-0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01],
            ParameterDescriptor(
                description="Convex polygon describing the part of the obstacle_frame frame that is visuable to the "
                "lidar of the robot. The other points are removed and therefore unseen to the robot. "
                "[x1, y1, x2, y2, ...]"
            ),
        )
        self.declare_parameter("publish_odometry", True)
        self.declare_parameter("obstacle_width", 0.1)

        road_width = 0.53

        ### Experiment 1: parked cars
        # distance_from_road = 0.15
        # length = 0.25
        # width = 0.15
        # distance_from_intersection = road_width + 0.5

        ### Experiment 2: big building
        distance_from_road = 10.0
        length = 3.0
        width = 3.0
        distance_from_intersection = road_width + distance_from_road

        static_obstacle = np.array(
            [
                [road_width + distance_from_road, -distance_from_intersection],
                [road_width + distance_from_road + width, -distance_from_intersection],
                [road_width + distance_from_road + width, -(distance_from_intersection + length)],
                [road_width + distance_from_road, -(distance_from_intersection + length)],
            ]
        )
        # static_obstacle2 = np.array(
        #     [
        #         [road_width + distance_from_road, -(distance_from_intersection + length + gap)],
        #         [road_width + distance_from_road + width, -(distance_from_intersection + length + gap)],
        #         [road_width + distance_from_road + width, -(distance_from_intersection + length + gap + length)],
        #         [road_width + distance_from_road, -(distance_from_intersection + length + gap + length)],
        #     ]
        # )
        # static_obstacle3 = static_obstacle2 - np.array([0, length + 0.1])
        # static_obstacle4 = static_obstacle3 - np.array([0, length + 0.1])

        self.declare_parameter(
            "static_obstacle",
            static_obstacle.flatten().tolist(),
            ParameterDescriptor(
                description="Static obstacle described by a list of consecutive corner points, which does not have to"
                " be convex: [x1, y1, x2, y2, ...]"
            ),
        )
        self.declare_parameter("only_visualize_static_obstacle", True)
        # self.declare_parameter("static_obstacle2", static_obstacle2.flatten().tolist())
        # self.declare_parameter("static_obstacle3", static_obstacle3.flatten().tolist())
        # self.declare_parameter("static_obstacle4", static_obstacle4.flatten().tolist())

        self.obstacle_frame = self.get_parameter("obstacle_frame").get_parameter_value().string_value
        self.scan_in_topic = self.get_parameter("scan_in_topic").get_parameter_value().string_value
        self.namespaces = self.get_parameter("namespaces").get_parameter_value().string_array_value
        self.scan_out_topic = self.get_parameter("scan_out_topic").get_parameter_value().string_value

        self.do_simulate_obstacles = self.get_parameter("do_simulate_obstacles").get_parameter_value().bool_value
        self.do_visualize = self.get_parameter("do_visualize").get_parameter_value().bool_value
        self.do_filter_scan = self.get_parameter("do_filter_scan").get_parameter_value().bool_value
        self.env_polygon = np.array(
            self.get_parameter("environment_boundary").get_parameter_value().double_array_value
        ).reshape(-1, 2)
        self.laser_filter_matrices = matrices_from_cw_cvx_polygon(self.env_polygon)
        self.do_publish_odometry = self.get_parameter("publish_odometry").get_parameter_value().bool_value
        self.obstacle_width = self.get_parameter("obstacle_width").get_parameter_value().double_value

        self.static_obstacles = []
        self.static_obstacles.append(
            np.array(self.get_parameter(f"static_obstacle").get_parameter_value().double_array_value).reshape(-1, 2)
        )
        self.only_visual_static_obs = self.get_parameter("only_visualize_static_obstacle").get_parameter_value().bool_value

        self.trajectories: dict[str, Trajectory] = {}

        self.create_subscription(LaserScan, self.scan_in_topic, self.laserscan_callback, 5)
        for namespace in self.namespaces:
            self.create_subscription(Trajectory, namespace + "/trajectory", self.get_trajectory_callback(namespace), 5)
        if self.do_publish_odometry:
            self.odometry_publishers = {
                n: self.create_publisher(Odometry, n + "/odometry/filtered", 5) for n in self.namespaces
            }
        self.scan_publisher = self.create_publisher(LaserScan, self.scan_out_topic, 5)

        # publish environment polygon
        if self.do_visualize:
            self.filter_polygon_pub = self.create_publisher(Marker, "visualization/road_env_polygon", 1)
            self.create_timer(5, self.visualize_filter_polygon_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # visualize static obstacles
        self.obstacles_visual_publisher = self.create_publisher(MarkerArray, "visualization/static_obstacles", 5)
        self.create_timer(3, self.static_obstacles_visualization_callback)

    def static_obstacles_visualization_callback(self) -> None:
        markers = []
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="planner")
        for idx, static_obstacle in enumerate(self.static_obstacles):
            x, y = static_obstacle[-1]
            point_type_list = [PointMsg(x=float(x), y=float(y))]
            for x, y in static_obstacle:
                point_type_list.append(PointMsg(x=float(x), y=float(y)))

            marker = Marker(
                header=header,
                id=idx,
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                points=point_type_list,
                scale=Vector3(x=0.03),
                color=ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),
                ns="static obstacles",
            )
            markers.append(marker)

        self.obstacles_visual_publisher.publish(MarkerArray(markers=markers))

    def visualize_filter_polygon_callback(self) -> None:
        """Visualize the filter polygon"""
        self.filter_polygon_pub.publish(self.points_to_linestrip_msg(self.env_polygon, "road env. filter"))

    def laserscan_callback(self, scan: LaserScan) -> None:
        """Callback function called for each received LaserScan message."""
        try:
            scan_filt = self.filter_laserscan(scan) if self.do_filter_scan else scan
            scan_filt_obst = self.add_obstacles_to_laserscan(scan_filt) if self.do_simulate_obstacles else scan_filt
            self.scan_publisher.publish(scan_filt_obst)
        except TransformException as ex:
            self.get_logger().info(str(ex))

    def filter_laserscan(self, scan: LaserScan) -> LaserScan:
        """Only keep measurements within a certain polygon. Other measurements are removed and default to the maximum
        view range."""
        points_laser_frame_3D = self.laserscan_to_points(scan, with_z=True)
        transform = self.tf_buffer.lookup_transform(
            self.obstacle_frame, scan.header.frame_id, Time.from_msg(scan.header.stamp)
        )
        points_obs_frame = transform_points(points_laser_frame_3D, transform.transform)[:, :2]
        A, B = self.laser_filter_matrices
        mask_filter = np.all(A @ points_obs_frame.T <= B.reshape(-1, 1), axis=0)
        ranges = np.array(scan.ranges)
        ranges[~mask_filter] = scan.range_max
        scan.ranges = ranges.tolist()

        return scan

    @staticmethod
    def laserscan_to_points(scan: LaserScan, with_z: bool = False) -> np.ndarray:
        """Simple convert a ROS LaserScan message to a array of ranges and angles. Also gives a mask of the invalid
        values and changes them to the view range to prevent math related errors or make to conservative decision when
        making the field of view."""
        ranges = np.array(scan.ranges)
        N = len(ranges)
        angles = scan.angle_min + np.arange(N) * scan.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        return np.hstack(((ranges * cos_sin_map).T, np.zeros((N, 1)))) if with_z else (ranges * cos_sin_map).T

    def get_trajectory_callback(self, namespace: str) -> Callable[[Trajectory], None]:
        def trajectory_callback(msg: Trajectory) -> None:
            self.trajectories[namespace] = msg
            return

        return trajectory_callback

    @staticmethod
    def stamp_to_float(stamp: TimeMsg) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def get_obstacle_odometry(trajectory: Trajectory, stamp: TimeMsg) -> Odometry:
        stamp_f = ScanSimulateNode.stamp_to_float(stamp)
        traject_stamps_f = [ScanSimulateNode.stamp_to_float(p.header.stamp) for p in trajectory.path.poses]

        # find the indices of the poses to interpolate between
        insert_idx = np.searchsorted(traject_stamps_f, stamp_f)
        next_idx, prev_idx = min(insert_idx, len(traject_stamps_f) - 1), max(0, insert_idx - 1)
        next_st, prev_st = traject_stamps_f[next_idx], traject_stamps_f[prev_idx]
        interp_factor = (stamp_f - prev_st) / (next_st - prev_st) if not next_st == prev_st else 0.0
        odometry = Odometry()
        pose = odometry.pose.pose
        odometry.header.frame_id = trajectory.path.header.frame_id
        odometry.header.stamp = stamp
        # interpolate the position
        next_position = trajectory.path.poses[next_idx].pose.position  # type: ignore
        prev_position = trajectory.path.poses[prev_idx].pose.position  # type: ignore
        pose.position.x = (1 - interp_factor) * prev_position.x + interp_factor * next_position.x
        pose.position.y = (1 - interp_factor) * prev_position.y + interp_factor * next_position.y

        # interpolate the orientation
        next_orient = [getattr(trajectory.path.poses[next_idx].pose.orientation, attr) for attr in ["x", "y", "z", "w"]]  # type: ignore
        prev_orient = [getattr(trajectory.path.poses[prev_idx].pose.orientation, attr) for attr in ["x", "y", "z", "w"]]  # type: ignore
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([prev_orient, next_orient]))
        obs_orient = slerp(interp_factor).as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = obs_orient

        # interpolate velocity
        if trajectory.velocities:
            next_velocity = trajectory.velocities[next_idx].linear.x  # type: ignore
            prev_velocity = trajectory.velocities[prev_idx].linear.x  # type: ignore
            odometry.twist.twist.linear.x = (1 - interp_factor) * prev_velocity + interp_factor * next_velocity

        return odometry

    def add_obstacles_to_laserscan(self, scan: LaserScan) -> LaserScan:
        """Adds the obstacles to the LaserScan message based on the trajectories published on the trajectory topics.
        Obstacle is assumed to be a circle, so the laser always see the same obstacle width."""
        # get the angles and ranges from the LaserScan message
        ranges = np.array(scan.ranges)
        N = len(ranges)
        angles = scan.angle_min + np.arange(N) * scan.angle_increment

        for namespace, trajectory in self.trajectories.items():
            # get the odometry of the simulated obstacle in the trajectory frame
            obstacle_odometry = self.get_obstacle_odometry(trajectory=trajectory, stamp=scan.header.stamp)
            if self.do_publish_odometry:
                self.odometry_publishers[namespace].publish(obstacle_odometry)

            # transform the odometry to the laser frame
            obs_pose_traj = PoseStamped(
                header=Header(stamp=scan.header.stamp, frame_id=trajectory.path.header.frame_id),
                pose=obstacle_odometry.pose.pose,
            )
            obs_pose_laser = self.tf_buffer.transform(obs_pose_traj, scan.header.frame_id)

            # adjust the ranges at the angles in the direction of the obstacle if the obstacle should be visible
            obs_x, obs_y = obs_pose_laser.pose.position.x, obs_pose_laser.pose.position.y  # type: ignore
            obstacle_angle = np.arctan2(obs_y, obs_x)
            obstacle_distance = np.hypot(obs_x, obs_y) - np.sqrt(2) * self.obstacle_width / 2  # closest point rectangle
            half_angle_width = np.arctan((self.obstacle_width / 2) / obstacle_distance)
            angle_mask = (angles >= obstacle_angle - half_angle_width) & (angles <= obstacle_angle + half_angle_width)
            ranges[angle_mask] = np.minimum(ranges[angle_mask], obstacle_distance)

        if not self.only_visual_static_obs:
            for obstacle_points in self.get_obstacles_in_laserframe(scan.header.frame_id):
                intersec_ranges = find_intersec_ranges(angles, obstacle_points, scan.range_max)
                ranges = np.minimum(intersec_ranges, ranges)

        ranges = np.maximum(ranges, 0.16)
        scan.ranges = ranges.tolist()
        return scan

    def get_obstacles_in_laserframe(self, laser_frame: str) -> list[np.ndarray]:
        static_obstacles_laser = []
        # self.get_logger().info(f"{self.static_obstacles=}")
        for obs_points in self.static_obstacles:
            transform = self.tf_buffer.lookup_transform(laser_frame, "planner", Time())
            obs_points_3D = np.hstack((obs_points, np.zeros((len(obs_points), 1))))
            # self.get_logger().info(f"{obs_points_3D=}")
            static_obstacles_laser.append(transform_points(obs_points_3D, transform.transform)[:, :2])

        # self.get_logger().info(f"{static_obstacles_laser=}")

        return static_obstacles_laser

    def points_to_linestrip_msg(self, points: np.ndarray, namespace: str, marker_idx: int = 0) -> Marker:
        """Convert a numpy array of points (shape = (N, 2)) to a ROS Marker message"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.obstacle_frame)

        point_type_list = [PointMsg(x=float(points[-1, 0]), y=float(points[-1, 1]))]
        for x, y in points:
            point_type_list.append(PointMsg(x=float(x), y=float(y)))

        marker = Marker(
            header=header,
            id=marker_idx,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            points=point_type_list,
            scale=Vector3(x=0.01),
            color=ColorRGBA(b=1.0, a=1.0),
            ns=namespace,
        )
        return marker


def main(args=None):
    rclpy.init(args=args)

    scan_simulate_node = ScanSimulateNode()

    rclpy.spin(scan_simulate_node)
    scan_simulate_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
