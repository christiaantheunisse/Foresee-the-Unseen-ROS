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
from typing import Tuple, List, Union, Optional, Literal

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point32, Point, Vector3, Quaternion, PolygonStamped, Polygon
from sensor_msgs.msg import LaserScan, PointCloud
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PolygonStamped  # necessary to enable Buffer.transform

from foresee_the_unseen.lib.helper_functions import matrix_from_transform, matrices_from_cw_cvx_polygon


def yaw_from_quaternion(quaternion: Quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


Direction = Literal["CW", "CCW"]


class FOVNode(Node):
    """This node deskews the Lidar scan and produces an underapproximating, closed FOV. This is achieved by correcting
    the Lidar measurements with an error model and merging multiple scans if necessary."""

    def __init__(self):
        super().__init__("scan_to_fov")

        self.scan_topic = "scan"
        self.deskewed_scan_topic = "scan/deskewed"
        self.filtered_scan_topic = "scan/road_env"
        self.fov_topic = "fov"
        self.filter_polygon_topic = "visualization/road_env_polygon"
        self.fov_frame = "map"
        self.do_visualize = True
        self.do_filter_scan = False
        self.rotating_direction: Direction = "CW"
        self.error_models_dir = "path/to/error_models"
        self.filter_polygon = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0]).reshape(-1, 2)
        self.view_range = 5.0  # TODO: get this from the configuration file

        assert self.rotating_direction == "CW", "The node is specificially made for a clockwise rotating node."

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 5)
        # self.merged_scan_pub = self.create_publisher(LaserScan, "/scan_merged", 5)
        self.deskewed_scan_pub = self.create_publisher(PointCloud, self.deskewed_scan_topic, 5)
        self.filtered_scan_pub = self.create_publisher(LaserScan, self.filtered_scan_topic, 5)
        self.fov_polygon_pub = self.create_publisher(PolygonStamped, self.fov_topic, 5)
        if self.do_visualize:
            self.filter_polygon_pub = self.create_publisher(Marker, self.filter_polygon_topic, 5)
            self.create_timer(5, self.visualize_filter_polygon_callback)

        # allow multithreading for the laser process functions
        laser_scan_process = ReentrantCallbackGroup()
        self.create_timer(1 / 100, self.process_scan, callback_group=laser_scan_process)

        self.laser_filter_matrices = matrices_from_cw_cvx_polygon(self.filter_polygon)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subsample_rate: Optional[int] = None
        self.subsample_rate: Optional[int] = 2
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

    def scan_callback(self, msg: LaserScan) -> None:
        self.last_scan = self.current_scan
        self.current_scan = msg
        self.scan_processed = False

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
            # self.merged_scan_pub.publish(new_scan)
            filtered_scan = self.filter_laserscan(new_scan) if self.do_filter_scan else new_scan

            ranges, angles, mask_invalid = self.laserscan_to_ranges_angles(
                filtered_scan, self.view_range, self.subsample_rate
            )
            ranges_corr, angles_corr = self.apply_error_model(ranges, angles, mask_invalid, 2)
            fov_points = self.make_fov(ranges_corr, angles_corr, mask_invalid)

            # Deskew the points constructing the FOV and publish the FOV
            fov_points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(filtered_scan, fov_points)
            fov_polygon_msg = self.points_to_polygon_msg(fov_points_deskewed, filtered_scan.header.stamp)
            self.fov_polygon_pub.publish(fov_polygon_msg)

            if self.do_visualize:
                # publish the deskewed pointcloud
                points = self.ranges_angles_to_points(ranges_corr, angles_corr)[~mask_invalid]
                points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(filtered_scan, points)
                pc_deskewed_msg = self.points_to_pointcloud_msg(points_deskewed, start_time)
                self.deskewed_scan_pub.publish(pc_deskewed_msg)

                # publish the filtered laserscan
                self.filtered_scan_pub.publish(filtered_scan)

            self.scan_processed = True
        except TransformException as ex:
            self.get_logger().info(str(ex))
            pass

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
        points_map = self.transform_points(
            points_laser, self.fov_frame, laser_frame, Time.from_msg(scan.header.stamp)
        )

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
        if delta_rotation <= 0.0:  # a value bigger than 0. means that there is a gap in the deskewed pointcloud
            return scan2
        else:
            if scan1.header.frame_id != scan2.header.frame_id:
                self.get_logger().error(
                    f"Both scans should be in the same frame: {scan1.header.frame_id=}, {scan2.header.frame_id=}"
                )
                raise TransformException
            # concetanate the last part of scan1
            N_to_concat = abs(math.ceil(delta_rotation / scan1.angle_increment))
            concat_header = Header(
                stamp=(start_time2 - Duration(seconds=N_to_concat * scan1.time_increment)).to_msg(),  # type: ignore
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
    ) -> np.ndarray:
        """Determine an underapproximating field of view from the LaserScan message."""
        smallest_range = ranges.min()
        start_idx = np.argmin(ranges)
        default_angle_incr = abs(np.abs(angles[1] - angles[0]))

        # step interpolate the range instead of linearly by always underapproximating
        #                  x--x                x--x
        # instead of:     /    \        do:    |  |
        #                /      \              |  |
        #               x        x          x---  ---x
        # x     == lidar measurement
        # -|/   == boundary of field of view (fov) constructed from these measurements
        ranges, angles = ranges[~mask_invalid], angles[~mask_invalid]
        ranges_extended = np.vstack((np.roll(ranges, 1), ranges, np.roll(ranges, -1))).T.flatten()
        is_bigger_than_prev = ranges > np.roll(ranges, 1)
        is_bigger_than_next = ranges > np.roll(ranges, -1)
        keep_original = np.ones_like(ranges, dtype=np.bool_)
        mask = np.vstack((is_bigger_than_prev, keep_original, is_bigger_than_next)).T.flatten()
        angles_extended = np.repeat(angles, 3)
        ranges, angles = ranges_extended[mask], angles_extended[mask]

        # calculate the ranges for an fov with a certain maximum inclination angles. This limits the maximum increase in
        #  the range between subsequent ranges.
        N = len(ranges)
        ranges = ranges.copy()
        for idx in np.roll(np.arange(N), -start_idx):  # [start_idx, start_idx+1, ..., N-1, 0, 1, ..., start_idx-1]
            current_range = ranges[idx]
            angle_incr = abs(angles[(idx + 1) % N] - angles[idx])
            angle_incr = default_angle_incr if angle_incr > 5 * default_angle_incr else angle_incr
            if angle_incr == 0.0:
                max_step = 0.0
            else:
                max_step = FOVNode.approx_max_step(current_range, max_incl_angle, angle_incr)
            ranges[(idx + 1) % N] = min(current_range + max_step, ranges[(idx + 1) % N])

        for idx in np.roll(np.arange(N), -start_idx)[::-1]:  # [start_idx-1, ..., 1, 0, N-1, N-2, ..., start_idx]
            current_range = ranges[idx]
            angle_incr = abs(angles[idx] - angles[(idx - 1) % N])
            angle_incr = default_angle_incr if angle_incr > 5 * default_angle_incr else angle_incr
            if angle_incr == 0.0:
                max_step = 0.0
            else:
                max_step = FOVNode.approx_max_step(current_range, max_incl_angle, angle_incr)
            ranges[(idx - 1) % N] = min(current_range + max_step, ranges[(idx - 1) % N])

        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points = (ranges * cos_sin_map).T  # 2D points

        return points

    def deskew_laserscan_interpolate_tf(
        self, scan: LaserScan, points: Optional[np.ndarray] = None
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
        f = interp1d(np.array([0, 1]), np.array([t_start_mat.flatten(), t_end_mat.flatten()]).T)
        t_interp_mats = f(np.linspace(0, 1, N)).reshape(4, 4, N).transpose((2, 0, 1))

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
        marker = self.points_to_linestrip_msg(self.filter_polygon, "Laser filter polygon")
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
