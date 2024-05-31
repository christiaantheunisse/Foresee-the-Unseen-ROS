import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import os
import numpy as np
import time
import copy
import math
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from typing import Tuple, List, Union, Optional, Literal, get_args

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point32, Point, Vector3, Quaternion, PolygonStamped, Polygon
from sensor_msgs.msg import LaserScan, PointCloud
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import ParameterDescriptor

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PolygonStamped  # necessary to enable Buffer.transform

from foresee_the_unseen.lib.helper_functions import matrix_from_transform, matrices_from_cw_cvx_polygon

from shapely.geometry import Polygon as ShapelyPolygon


def yaw_from_quaternion(quaternion: Quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


Direction = Literal["CW", "CCW"]


def on_segment(p, q, r) -> bool:
    """all inputs are points with shape (2,). The function checks if point q lies on line segment 'pr'."""
    return (
        (q[0] <= max(p[0], r[0]))
        and (q[0] >= min(p[0], r[0]))
        and (q[1] <= max(p[1], r[1]))
        and (q[1] >= min(p[1], r[1]))
    )


def orientation(p, q, r) -> Literal[0, 1, 2]:
    """all inputs are points with shape (2,)
    function returns the following values:
        0 : Collinear points
        1 : Clockwise points
        2 : Counterclockwise
    """
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        # Clockwise orientation
        return 1
    elif val < 0:
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


def do_intersect(p1, q1, p2, q2) -> bool:
    """All inputs are points with shape (2,).
    The function returns true if the line segments 'p1q1' and 'p2q2' intersect."""
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True
    # If none of the cases
    return False


def ccw(A, B, C):
    """All inputs are points with shape (2,)"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def do_intersect_simple(A, B, C, D):
    """All inputs are points with shape (2,).
    The function returns true if the line segments 'AB' and 'CD' intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def check_self_intersection_polygon(points: np.ndarray) -> np.ndarray:
    """Check if the polygon intersects itself"""
    line_segments = np.stack((points, np.roll(points, -1, axis=0)), axis=1)
    N = len(line_segments)
    masks = []
    for idx, segment in enumerate(line_segments):
        mask = [
            (
                do_intersect(segment[0], segment[1], ls[0], ls[1])
                if idx_sub not in [(idx - 1) % N, idx, (idx + 1) % N]
                else False
            )
            for idx_sub, ls in enumerate(line_segments)
        ]
        masks.append(mask)
    masks = np.array(masks)
    # return line_segments[masks.sum(axis=1) > 0]
    return np.array(masks)


def get_self_intersections_polygon(points: np.ndarray, no_of_neighbours_to_check: int = 6) -> np.ndarray:
    """Gives an array of combinations of line segments that intersect: [[line_segment_i, line_segment_j], ...]
    Only checks at a limited number of neighbours"""
    line_segments = np.stack((points, np.roll(points, -1, axis=0)), axis=1)
    N_line_segments = len(line_segments)
    intersections = []
    no_of_neighbours_to_check = 6
    check_idxs = np.arange(2, no_of_neighbours_to_check + 1)
    for idx_a, segment_a in enumerate(line_segments):
        for idx_b in (check_idxs + idx_a) % N_line_segments:
            segment_b = line_segments[idx_b]
            if do_intersect_simple(segment_a[0], segment_a[1], segment_b[0], segment_b[1]):
                intersections.append([idx_a, idx_b])
    if len(intersections):
        return np.sort(intersections, axis=1)  # [[ls_i, ls_j], ...]
    else:
        return np.array(intersections)


def make_valid_polygon(points: np.ndarray) -> np.ndarray:
    """Recursively remove self-intersections from a polygon"""
    for i in range(10):
        N = len(points)
        mask_intersections = check_self_intersection_polygon(points)
        intersections = np.array(np.where(mask_intersections)).T  # [[ls_i, ls_j], ...]
        intersections = get_self_intersections_polygon(points)  # [[ls_i, ls_j], ...]
        if not len(intersections):
            return points

        # just treat the first intersection, until no intersections are left:
        intersect = intersections[0]
        #  remove the lowest possible number of intermediate points
        assert np.diff(intersect) > 0, f"{intersect=} {intersections=}"
        if np.diff(intersect) < N / 2:
            points = np.vstack((points[: intersect[0] + 1], points[intersect[1] :]))
        else:
            points = points[intersect[0] : intersect[1] + 1]

    raise StopIteration


class FOVNode(Node):
    """This node deskews the Lidar scan and produces an underapproximating, closed FOV. This is achieved by correcting
    the Lidar measurements with an error model and merging multiple scans if necessary."""

    def __init__(self):
        super().__init__("scan_to_fov")

        self.declare_parameter("fov_frame", "map")

        self.declare_parameter("scan_topic", "scan")
        self.declare_parameter("deskewed_scan_topic", "scan/deskewed")
        self.declare_parameter("environment_scan_topic", "scan/road_env")
        self.declare_parameter("fov_topic", "fov")
        self.declare_parameter("environment_polygon_topic", "visualization/road_env_polygon")

        self.declare_parameter("do_visualize", True)
        self.declare_parameter("do_filter_scan", False)

        self.declare_parameter("rotating_direction", "CW")
        self.declare_parameter("view_range", 5.0)
        self.declare_parameter("error_models_directory", "path/to/error_models")
        self.declare_parameter(
            "environment_boundary",
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ParameterDescriptor(
                description="Convex polygon describing the part of the fov frame that is visuable to the lidar of the"
                " robot. The other points are removed and therefore unseen to the robot. [x1, y1, x2, y2, ...]"
            ),
        )
        self.fov_frame = self.get_parameter("fov_frame").get_parameter_value().string_value

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.deskewed_scan_topic = self.get_parameter("deskewed_scan_topic").get_parameter_value().string_value
        self.env_scan_topic = self.get_parameter("environment_scan_topic").get_parameter_value().string_value
        self.fov_topic = self.get_parameter("fov_topic").get_parameter_value().string_value
        self.env_polygon_topic = self.get_parameter("environment_polygon_topic").get_parameter_value().string_value

        self.do_visualize = self.get_parameter("do_visualize").get_parameter_value().bool_value
        self.do_filter_scan = self.get_parameter("do_filter_scan").get_parameter_value().bool_value

        self.rotating_direction = self.get_parameter("rotating_direction").get_parameter_value().string_value
        assert self.rotating_direction in get_args(
            Direction
        ), f"`rotating_direction` should have one of the following values: {get_args(Direction)}"
        self.view_range = self.get_parameter("view_range").get_parameter_value().double_value
        self.error_models_dir = self.get_parameter("error_models_directory").get_parameter_value().string_value
        self.env_polygon = np.array(
            self.get_parameter("environment_boundary").get_parameter_value().double_array_value
        ).reshape(-1, 2)

        assert self.rotating_direction == "CW", "The node is specificially made for a clockwise rotating node."

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 5)
        self.deskewed_scan_pub = self.create_publisher(PointCloud, self.deskewed_scan_topic, 5)
        self.environment_scan_pub = self.create_publisher(LaserScan, self.env_scan_topic, 5)
        self.fov_polygon_pub = self.create_publisher(PolygonStamped, self.fov_topic, 5)
        if self.do_visualize:
            self.filter_polygon_pub = self.create_publisher(Marker, self.env_polygon_topic, 5)
            self.create_timer(5, self.visualize_filter_polygon_callback)

        # allow multithreading for the laser process functions
        laser_scan_process = ReentrantCallbackGroup()
        self.create_timer(1 / 100, self.process_scan, callback_group=laser_scan_process)

        self.laser_filter_matrices = matrices_from_cw_cvx_polygon(self.env_polygon)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.subsample_rate: Optional[int] = None
        self.subsample_rate: Optional[int] = 20

        self.scan_processed: bool = True
        self.last_scan: Optional[LaserScan] = None
        self.current_scan: LaserScan = LaserScan()

        # TODO: load the right error models
        # define some dummy error models:
        #   Lidar error models are only dependent on the measured range, so 1D
        # angle error model
        no_samples = 1000
        input_data_angle_xs = np.linspace(0.15, 6, no_samples)  # the ranges
        input_data_angle_means = np.linspace(-0.005, -0.015, no_samples)
        input_data_angle_stds = np.linspace(0.005, 0.005, no_samples)
        self.angle_error_model_mean = interp1d(input_data_angle_xs, input_data_angle_means, fill_value="extrapolate")
        self.angle_error_model_std = interp1d(input_data_angle_xs, input_data_angle_stds, fill_value="extrapolate")
        # range error model
        no_samples = 1000
        input_data_range_xs = np.linspace(0.15, 6, no_samples)  # the ranges
        input_data_range_means = np.linspace(0.002, 0.08, no_samples)
        input_data_range_stds = np.linspace(0.02, 0.06, no_samples)
        self.range_error_model_mean = interp1d(input_data_range_xs, input_data_range_means, fill_value="extrapolate")
        self.range_error_model_std = interp1d(input_data_range_xs, input_data_range_stds, fill_value="extrapolate")

        self.counter = 0

    def scan_callback(self, msg: LaserScan) -> None:
        self.last_scan = self.current_scan
        self.current_scan = msg
        self.scan_processed = False
        self.counter += 1

    def process_scan(self) -> None:
        """The scan callback"""
        # Check if a new scan is available
        if self.scan_processed:
            return

        # check if the transform for the last lidar point is already available
        msg = self.current_scan
        start_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        end_time = start_time + Duration(seconds=msg.scan_time)  # type: ignore
        if not self.tf_buffer.can_transform(self.fov_frame, msg.header.frame_id, end_time) or self.last_scan is None:
            return

        exec_start_time = time.time()
        try:
            new_scan = self.merge_scans(self.last_scan, self.current_scan)
            filtered_scan = self.filter_laserscan(new_scan) if self.do_filter_scan else new_scan

            ranges, angles, mask_invalid = self.laserscan_to_ranges_angles(
                filtered_scan, self.view_range, self.subsample_rate
            )
            ranges_corr, angles_corr = self.apply_error_model(ranges, angles, mask_invalid, 2)
            fov_points, time_order = self.make_fov(ranges_corr, angles_corr, mask_invalid)

            # Deskew the points defining the FOV and publish the FOV
            fov_points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(
                filtered_scan, fov_points, time_order
            )
            try:
                fov_points_deskewed = make_valid_polygon(fov_points_deskewed)
            except StopIteration:
                self.get_logger().error("Making polygon valid failed")
            if not ShapelyPolygon(fov_points_deskewed).is_valid:
                self.get_logger().error("Invalid polygon!")
            fov_polygon_msg = self.points_to_polygon_msg(fov_points_deskewed, filtered_scan.header.stamp)
            self.fov_polygon_pub.publish(fov_polygon_msg)

            if self.do_visualize:
                # publish the deskewed pointcloud
                points = self.ranges_angles_to_points(ranges_corr, angles_corr)[~mask_invalid]
                points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(filtered_scan, points)
                # pc_deskewed_msg = self.points_to_pointcloud_msg(points_deskewed, start_time)
                pc_deskewed_msg = self.points_to_pointcloud_msg(fov_points_deskewed, start_time)
                self.deskewed_scan_pub.publish(pc_deskewed_msg)

                # publish the filtered laserscan
                self.environment_scan_pub.publish(filtered_scan)

            self.scan_processed = True
        except TransformException as ex:
            self.get_logger().info(str(ex))
            pass

        print(f"Execution time is: {(time.time() - exec_start_time) * 1000:.0f} ms")

    @staticmethod
    def laserscan_to_ranges_angles(
        scan: LaserScan,
        view_range: float,
        subsample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple convert a ROS LaserScan message to a array of ranges and angles. Also gives a mask of the invalid
        values and changes them to the view range to prevent math related errors or make to conservative decision when
        making the field of view."""
        ranges = np.array(scan.ranges)
        if subsample_rate:
            ranges = np.resize(ranges, (math.ceil(len(ranges) / subsample_rate), subsample_rate)).min(axis=1)
            ranges = np.append(ranges, scan.ranges[-1])
        # mask_invalid = (ranges < scan.range_min) | (ranges > scan.range_max)
        mask_invalid = ranges < scan.range_min
        ranges[mask_invalid | (ranges > view_range)] = view_range
        N = len(ranges)
        if subsample_rate:
            angles = scan.angle_min + np.arange(N - 1) * scan.angle_increment * (
                subsample_rate if subsample_rate else 1
            )
            angles = np.append(angles, scan.angle_max)
        else:
            angles = scan.angle_min + np.arange(N) * scan.angle_increment * (subsample_rate if subsample_rate else 1)

        return ranges, angles, mask_invalid

    @staticmethod
    def ranges_angles_to_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        return (ranges * cos_sin_map).T

    @staticmethod
    def laserscan_to_points(scan: LaserScan, view_range: float) -> np.ndarray:
        """Simple convert a ROS LaserScan message to a array of points"""
        ranges, angles, mask = FOVNode.laserscan_to_ranges_angles(scan, view_range)
        return FOVNode.ranges_angles_to_points(ranges[~mask], angles[~mask])

    def filter_laserscan(self, scan: LaserScan) -> LaserScan:
        """Filter points from the pointcloud from the lidar to reduce the computation of the `datmo` package."""
        # Convert the ranges to a pointcloud
        laser_frame = scan.header.frame_id
        ranges, angles, _ = self.laserscan_to_ranges_angles(scan, self.view_range)
        points_laser = self.ranges_angles_to_points(ranges, angles)
        points_map = self.transform_points(points_laser, self.fov_frame, laser_frame, Time.from_msg(scan.header.stamp))

        # filter the points based on a polygon
        A, B = self.laser_filter_matrices
        # mask_polygon = np.all(A @ points_map.T <= np.repeat(B.reshape(-1, 1), points_map.shape[0], axis=1), axis=0)
        mask_polygon = np.all(A @ points_map.T <= B.reshape(-1, 1), axis=0)

        # filter the points based on a circle -> distance from origin
        mask_fov = np.array(scan.ranges) <= self.view_range

        # Combine masks and make new message
        mask = mask_polygon & mask_fov
        new_msg = scan
        ranges = np.array(scan.ranges).astype(np.float32)
        ranges[~mask] = np.inf  # set filtered out values to infinite
        new_msg.ranges = ranges

        return new_msg

    def merge_scans(self, scan1: LaserScan, scan2: LaserScan) -> LaserScan:
        """Merge two laserscans to account for the gap in the LaserScan that will occur in some situations when the
        LaserScan is deskewed."""
        start_time2 = Time.from_msg(scan2.header.stamp)
        end_time2 = start_time2 + Duration(seconds=scan2.scan_time)  # type: ignore
        laser_frame = scan2.header.frame_id

        t_end2 = self.tf_buffer.lookup_transform(self.fov_frame, laser_frame, end_time2)
        t_start2 = self.tf_buffer.lookup_transform(self.fov_frame, laser_frame, start_time2)  # type: ignore
        delta_rotation = yaw_from_quaternion(t_end2.transform.rotation) - yaw_from_quaternion(
            t_start2.transform.rotation
        )
        # if delta_rotation <= 0.0:  # a value bigger than 0. means that there is a gap in the deskewed pointcloud
        if (
            delta_rotation <= scan1.angle_increment
        ):  # a value bigger than 0. means that there is a gap in the deskewed pointcloud
            # remove points from the scan
            N_to_delete = abs(math.ceil(delta_rotation / scan2.angle_increment))
            new_start_time2 = start_time2 + Duration(seconds=N_to_delete * scan2.time_increment)  # type: ignore
            # print(new_start_time2.nanoseconds // 1e9, new_start_time2.nanoseconds % 1e9)
            scan2.header.stamp = new_start_time2.to_msg()
            scan2.angle_min = scan2.angle_min + N_to_delete * scan2.angle_increment
            scan2.scan_time = scan2.scan_time - scan2.time_increment * N_to_delete
            scan2.ranges = scan2.ranges[N_to_delete:]

            return scan2
        else:
            if scan1.header.frame_id != scan2.header.frame_id:
                self.get_logger().error(
                    f"Both scans should be in the same frame: {scan1.header.frame_id=}, {scan2.header.frame_id=}"
                )
                raise TransformException
            # concetanate the last part of scan1
            # N_to_concat = abs(math.ceil(delta_rotation / scan1.angle_increment))
            N_to_concat = abs(int(delta_rotation / scan1.angle_increment))
            concat_header = Header(
                stamp=(start_time2 - Duration(seconds=N_to_concat * scan2.time_increment)).to_msg(),  # type: ignore
                frame_id=scan2.header.frame_id,
            )
            concat_ranges = scan1.ranges[-N_to_concat:] + scan2.ranges
            concat_scan = LaserScan(
                header=concat_header,
                angle_min=scan2.angle_min - N_to_concat * scan2.angle_increment,
                angle_max=scan2.angle_max,
                angle_increment=scan2.angle_increment,
                time_increment=scan2.time_increment,
                scan_time=scan2.time_increment * (len(concat_ranges) - 1),
                range_min=scan2.range_min,
                range_max=scan2.range_max,
                ranges=concat_ranges,
            )
            return concat_scan

    def apply_error_model(
        self, ranges: np.ndarray, angles: np.ndarray, mask_invalid: np.ndarray, stds_margin: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the error model on the measured ranges and angles:
            range error = an deviation from the true range of the measured range value
            angle error = an deviation from the true angle in the angle at which a certain range is measured.
        To apply the range error model:
            Simply reduce the measured ranges
        To apply the angle error model:
            Map the measured range at each angle over a sweep of angles in both angular directions by the error size.
            Choose the closest range at each angle from the overlapping sweeps.
        """
        range_errors = self.range_error_model_mean(ranges) + stds_margin * self.range_error_model_std(ranges)
        # angle error should be strictly positive: negative error means that obstacles are lost
        angle_errors = np.minimum(
            self.angle_error_model_mean(ranges) + stds_margin * self.angle_error_model_std(ranges), 0
        )
        ranges = ranges - range_errors
        angle_increment = np.abs(angles[1] - angles[0])
        angle_error_steps = np.ceil(angle_errors / angle_increment).astype(np.int_)
        angle_error_steps[mask_invalid] = 0
        N = len(ranges)
        new_ranges = ranges.copy()
        for idx, steps in enumerate(angle_error_steps):
            idcs = np.arange(idx - steps, idx + steps + 1) % N  # to wrap the indices to a circulair array
            new_ranges[idcs] = np.minimum(new_ranges[idcs], ranges[idx])
        return new_ranges, angles

    @staticmethod
    def approx_max_step(distance: float, max_incl_angle: float, angle_incr: float) -> float:
        """Approximate the maximum increase in range based on the current range and the maximum inclination angle."""
        opposite = np.sin(angle_incr) * distance
        return opposite / np.tan(np.pi / 2 - max_incl_angle)

    @staticmethod
    def make_fov(
        ranges: np.ndarray,
        angles: np.ndarray,
        mask_invalid: np.ndarray,
        max_incl_angle: float = np.radians(70),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determine an underapproximating field of view from the LaserScan message."""
        start_idx = np.argmin(ranges)
        default_angle_incr = abs(np.abs(angles[1] - angles[0]))

        # step interpolate the range instead of linearly by always underapproximating
        #                  x--x                x--x
        # instead of:     /    \        do:    |  |
        #                /      \              |  |
        #               x        x          x---  ---x
        # x     == lidar measurement
        # -|/   == boundary of field of view (fov) constructed from these measurements
        time_order = np.arange(len(ranges))
        ranges, angles, time_order = ranges[~mask_invalid], angles[~mask_invalid], time_order[~mask_invalid]

        # calculate the ranges for an fov with a certain maximum inclination angles. This limits the maximum increase in
        #  the range between subsequent ranges.
        N = len(ranges)
        ranges = ranges.copy()
        for idx in np.roll(np.arange(N), -start_idx):  # [start_idx, start_idx+1, ..., N-1, 0, 1, ..., start_idx-1]
            current_range = ranges[idx]
            angle_incr = angles[(idx + 1) % N] - angles[idx] if (idx + 1) != N else default_angle_incr
            max_step = FOVNode.approx_max_step(current_range, max_incl_angle, angle_incr)
            ranges[(idx + 1) % N] = min(current_range + max_step, ranges[(idx + 1) % N])

        for idx in np.roll(np.arange(N), -start_idx)[::-1]:  # [start_idx-1, ..., 1, 0, N-1, N-2, ..., start_idx]
            current_range = ranges[idx]
            angle_incr = angles[idx] - angles[(idx - 1) % N] if idx != 0 else default_angle_incr
            max_step = FOVNode.approx_max_step(current_range, max_incl_angle, angle_incr)
            ranges[(idx - 1) % N] = min(current_range + max_step, ranges[(idx - 1) % N])

        ranges_extended = np.vstack((np.roll(ranges, 1), ranges, np.roll(ranges, -1))).T.flatten()
        is_bigger_than_prev = ranges > np.roll(ranges, 1)
        is_bigger_than_next = ranges > np.roll(ranges, -1)
        keep_original = ~(is_bigger_than_prev & is_bigger_than_next)
        mask = np.vstack((is_bigger_than_prev, keep_original, is_bigger_than_next)).T.flatten()
        angles_extended = np.repeat(angles, 3)
        time_order_extended = np.repeat(np.arange(len(ranges)), 3)
        ranges, angles, time_order = ranges_extended[mask], angles_extended[mask], time_order_extended[mask]

        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points = (ranges * cos_sin_map).T  # 2D points

        return points, time_order

    def deskew_laserscan_interpolate_tf(
        self, scan: LaserScan, points: Optional[np.ndarray] = None, time_order: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Time]:
        """Deskew a ROS Laserscan using /tf. The transforms at the beginning and the end of the scan is obtained and are
        linearly interpolated for the intermediate points."""
        start_time = Time.from_msg(scan.header.stamp)
        end_time = start_time + Duration(seconds=scan.scan_time)  # type: ignore
        laser_frame = scan.header.frame_id

        t_end = self.tf_buffer.lookup_transform(self.fov_frame, laser_frame, end_time)
        t_start = self.tf_buffer.lookup_transform(self.fov_frame, laser_frame, start_time)  # type: ignore
        t_start_mat, t_end_mat = matrix_from_transform(t_start), matrix_from_transform(t_end)

        points = self.laserscan_to_points(scan, self.view_range) if points is None else points
        # invert the point order if necessary
        points = np.flip(points, axis=0) if self.rotating_direction == "CW" else points
        points_4D = np.hstack((points, np.zeros((len(points), 1)), np.ones((len(points), 1))))

        # linearly interpolate the transformation matrix
        N = len(points)
        if time_order is None:
            time_order = np.linspace(0, 1, N)
            value_range = [0, 1]
        else:
            value_range = np.array([time_order.min(), time_order.max()])
        f = interp1d(value_range, np.array([t_start_mat.flatten(), t_end_mat.flatten()]).T)
        t_interp_mats = f(time_order).reshape(4, 4, N).transpose((2, 0, 1))

        # VERY ugly method
        points_deskewed = []
        for t, p in zip(t_interp_mats, points_4D):
            points_deskewed.append((t @ p)[:2])
        points_deskewed = np.array(points_deskewed)

        return points_deskewed, start_time

    def transform_points(
        self, points: np.ndarray, target_frame: str, source_frame: str, time_stamp: Time
    ) -> np.ndarray:
        """Converts a pointcloud of 2D points (np.ndarray with shape = (N, 2)) with a ROS transform.
        Raises a TransformException when the transform is not available"""
        assert points.shape[1] == 2, "Should be 2D points with array shape (N, 2)"
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, time_stamp)
        points_4D = np.hstack((points, np.zeros((len(points), 1)), np.ones((len(points), 1))))
        t_mat = matrix_from_transform(t)
        return (t_mat @ points_4D.T)[:2].T

    def visualize_filter_polygon_callback(self) -> None:
        """Visualize the polygon to filter the pointcloud resulting from the lidar scan"""
        marker = self.points_to_linestrip_msg(self.env_polygon, "Laser filter polygon")
        self.filter_polygon_pub.publish(marker)

    def points_to_pointcloud_msg(self, points: np.ndarray, start_time: Time) -> PointCloud:
        """Convert a numpy array of points (shape = (N, 2)) to a ROS PointCloud message"""
        header = Header(stamp=start_time.to_msg(), frame_id=self.fov_frame)
        point32_list = [Point32(x=p[0], y=p[1]) for p in points.astype(np.float_)]
        return PointCloud(header=header, points=point32_list)

    def points_to_linestrip_msg(self, points: np.ndarray, namespace: str, marker_idx: int = 0) -> Marker:
        """Convert a numpy array of points (shape = (N, 2)) to a ROS Marker message"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.fov_frame)

        point_type_list = [Point(x=float(points[-1, 0]), y=float(points[-1, 1]))]
        for x, y in points:
            point_type_list.append(Point(x=float(x), y=float(y)))

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

    def points_to_polygon_msg(self, points: np.ndarray, stamp: TimeMsg) -> PolygonStamped:
        """Convert a numpy array of points (shape = (N, 2)) to a ROS PolygonStamped message"""
        header = Header(stamp=stamp, frame_id=self.fov_frame)
        # point_type_list = [Point32(x=float(points[-1, 0]), y=float(points[-1, 1]))]
        point32_list: List[Point32] = []
        for x, y in points:
            point32_list.append(Point32(x=float(x), y=float(y)))

        return PolygonStamped(header=header, polygon=Polygon(points=point32_list))


def main(args=None):
    rclpy.init(args=args)

    fov_node = FOVNode()

    rclpy.spin(fov_node)
    fov_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
