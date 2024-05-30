import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.node import Node
import os
import numpy as np
import time
import copy
import math
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from typing import Tuple, List, Union, Optional, Literal

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point32, Point, Vector3
from sensor_msgs.msg import LaserScan, PointCloud
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import matrix_from_transform, euler_from_quaternion

Direction = Literal["CW", "CCW"]

def approx_max_step(distance: float, max_incl_angle: float, angle_incr: float) -> float:
    opposite = np.sin(angle_incr) * distance
    return opposite / np.tan(np.pi / 2 - max_incl_angle)

def make_fov(
    ranges: np.ndarray, angles: np.ndarray, mask_invalid: np.ndarray, max_incl_angle: float = np.radians(70), subsample_rate: int = 5
) -> np.ndarray:
    start_idx = np.argmin(ranges)
    angle_increment = np.abs(angles[1] - angles[0])

    # step interpolate the range instead of linearly
    ranges, angles = ranges[~mask_invalid], angles[~mask_invalid]
    ranges_extended = np.vstack((np.roll(ranges, 1), ranges, np.roll(ranges, -1))).T.flatten()
    is_bigger_than_prev = ranges > np.roll(ranges, 1)
    is_bigger_than_next = ranges > np.roll(ranges, -1)
    keep_original = np.ones_like(ranges, dtype=np.bool_)
    mask = np.vstack((is_bigger_than_prev, keep_original, is_bigger_than_next)).T.flatten()
    angles_extended = np.repeat(angles, 3)
    ranges, angles = ranges_extended[mask], angles_extended[mask]

    # calculate the maximum range step
    N = len(ranges)
    ranges = ranges.copy()
    for idx in np.roll(np.arange(N), -start_idx):
        current_range = ranges[idx]
        angle_increment = angles[(idx+1)%N] - angles[idx]
        if angle_increment == 0.:
            max_step = 0.
        else:
            max_step = approx_max_step(current_range, max_incl_angle, angle_increment)
        ranges[(idx + 1) % N] = min(current_range + max_step, ranges[(idx + 1) % N])

    for idx in np.roll(np.arange(N), -start_idx)[::-1]:
        current_range = ranges[idx]
        angle_increment = angles[idx] - angles[(idx - 1)%N]
        if angle_increment == 0.:
            max_step = 0.
        else:
            max_step = approx_max_step(current_range, max_incl_angle, angle_increment)
        ranges[(idx - 1) % N] = min(current_range + max_step, ranges[(idx - 1) % N])


    # ranges = np.vstack((ranges, np.minimum(np.roll(ranges, 1), ranges, np.roll(ranges, -1)))).T.flatten()
    # ranges = np.vstack((ranges, np.roll(ranges, -1))).T.flatten()
    # angles = np.repeat(angles, 2)
    cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
    points = (ranges * cos_sin_map).T  # 2D points

    return points

class DeskewLidarNode(Node):
    def __init__(self):
        super().__init__("deskew_lidar")

        self.scan_topic = "scan"
        self.skewed_scan_topic = "skewed_scan"
        self.deskewed_scan_topic = "deskewed_scan"
        self.fov_visualization_topic = "visualization/fov"
        self.deskew_frame = "map"
        self.scan_frequency = 8
        self.scan_time = 1 / self.scan_frequency
        self.rotating_direction: Direction = "CW"

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 5)
        self.deskewed_scan_pub = self.create_publisher(PointCloud, self.deskewed_scan_topic, 5)
        self.skewed_scan_pub = self.create_publisher(PointCloud, self.skewed_scan_topic, 5)
        self.fov_pub = self.create_publisher(Marker, self.fov_visualization_topic, 5)

        # self.create_subscription(Odometry, "odometry/filtered", self.odometry_callback, 5)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subsample_rate: Optional[int] = None
        self.subsample_rate: Optional[int] = 5
        self.last_scan: Optional[LaserScan] = None

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

    def odometry_callback(self, msg: Odometry) -> None:
        """Just check the clock performance"""
        clock_time_float = self.get_clock().now().nanoseconds * 1e-9
        msg_time_float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        print(f"{clock_time_float - msg_time_float=}, {(clock_time_float - msg_time_float) > 0}")

    def scan_callback(self, msg: LaserScan) -> None:
        """The scan callback"""
        exec_start_time = time.time()
        if self.last_scan is None:
            self.last_scan = msg
            return
        try:
            # Apply the error models
            for idx, scan in enumerate([self.last_scan, msg]):
                ranges, angles, mask_invalid = self.laserscan_to_ranges_angles(scan, self.subsample_rate)
                ranges_corr, angles_corr = self.apply_error_model(ranges, angles, mask_invalid, 2)
                fov_points = make_fov(ranges_corr, angles_corr, mask_invalid)
                points = self.ranges_angles_to_points(ranges_corr, angles_corr)[~mask_invalid]

                # Deskew by INTERPOLATING tfs
                points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(scan, points)
                fov_points_deskewed, start_time = self.deskew_laserscan_interpolate_tf(scan, fov_points)

                # publish the deskewed pointcloud
                pc_deskewed_msg = self.points_to_pointcloud_msg(points_deskewed, start_time)
                self.deskewed_scan_pub.publish(pc_deskewed_msg)

                # publish the fov
                fov_marker_msg = self.points_to_linestrip_msg(fov_points_deskewed, f"fov_{idx}")
                self.fov_pub.publish(fov_marker_msg)

            # add minimum range 15cm polygon
            # msg_min_range = copy.deepcopy(msg)
            # msg_min_range.ranges = [msg.range_min for _ in range(len(msg.ranges))]
            # ranges_min_range, angles_min_range, _ = self.laserscan_to_ranges_angles(msg_min_range, 20)
            # points_min_range = self.ranges_angles_to_points(ranges_min_range, angles_min_range)
            # points_min_range_desk, start_time = self.deskew_laserscan_interpolate_tf(msg_min_range, points_min_range)
            # min_range_marker_msg = self.points_to_linestrip_msg(points_min_range_desk, "min_range")
            # self.fov_pub.publish(min_range_marker_msg)

            # Deskew by SAMPLING tfs
            # points, start_time = self.deskew_laserscan_sample_tf(msg)
            # pointcloud_deskewed_msg = self.points_to_pointcloud_msg(points, start_time)
            # # pointcloud_deskewed_msg = self.points_to_pointcloud_msg(points, Time())
            # self.deskewed_scan_publisher.publish(pointcloud_deskewed_msg)

            # Don't deskew at all
            points_skewed_laser_frame = self.laserscan_to_points(msg)
            start_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
            points_skewed = self.transform_pointcloud(
                points_skewed_laser_frame, self.deskew_frame, msg.header.frame_id, start_time
            )
            pointcloud_skewed_msg = self.points_to_pointcloud_msg(points_skewed, start_time)
            self.skewed_scan_pub.publish(pointcloud_skewed_msg)

        except TransformException as ex:
            print(ex)
            pass

        self.last_scan = msg
        print(f"Total execution time = {(time.time() - exec_start_time) * 1000:.0f} ms")

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
    def laserscan_to_ranges_angles(
        scan: LaserScan, subsample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple convert a ROS LaserScan message to a array of ranges and angles"""
        ranges = np.array(scan.ranges)
        if subsample_rate:
            ranges = np.resize(ranges, (math.ceil(len(ranges) / subsample_rate), subsample_rate)).min(axis=1)
            ranges = np.append(ranges, scan.ranges[-1])
        mask_invalid = (ranges < scan.range_min) | (ranges > scan.range_max)
        ranges[mask_invalid] = scan.range_max
        N = len(ranges)
        if subsample_rate:
            angles = scan.angle_min + np.arange(N-1) * scan.angle_increment * (subsample_rate if subsample_rate else 1)
            angles = np.append(angles, scan.angle_max)
        else:
            angles = scan.angle_min + np.arange(N) * scan.angle_increment * (subsample_rate if subsample_rate else 1)

        return ranges, angles, mask_invalid

    @staticmethod
    def ranges_angles_to_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        return (ranges * cos_sin_map).T

    @staticmethod
    def laserscan_to_points(scan: LaserScan) -> np.ndarray:
        """Simple convert a ROS LaserScan message to a array of points"""
        ranges, angles, mask = DeskewLidarNode.laserscan_to_ranges_angles(scan)
        return DeskewLidarNode.ranges_angles_to_points(ranges[~mask], angles[~mask])

    def deskew_laserscan_interpolate_tf(
        self, scan: LaserScan, points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Time]:
        """Deskew a ROS Laserscan by interpolating the transform"""
        start_time = Time(seconds=scan.header.stamp.sec, nanoseconds=scan.header.stamp.nanosec)
        end_time = start_time + Duration(seconds=scan.scan_time)  # type: ignore
        laser_frame = scan.header.frame_id

        # tf data is often not available for the end time (often about a few ms)
        # Get latest and subtract the total scan_time
        use_latest = False
        if use_latest:
            t_end = self.tf_buffer.lookup_transform(self.deskew_frame, laser_frame, Time())
            t_end_time = Time(seconds=t_end.header.stamp.sec, nanoseconds=t_end.header.stamp.nanosec)
            t_start = self.tf_buffer.lookup_transform(self.deskew_frame, laser_frame, t_end_time - Duration(seconds=scan.scan_time))  # type: ignore
            # t_start_time = Time(seconds=t_start.header.stamp.sec, nanoseconds=t_start.header.stamp.nanosec)
        else:
            t_end = self.tf_buffer.lookup_transform(self.deskew_frame, laser_frame, end_time)
            # t_end_time = Time(seconds=t_end.header.stamp.sec, nanoseconds=t_end.header.stamp.nanosec)
            t_start = self.tf_buffer.lookup_transform(self.deskew_frame, laser_frame, start_time)  # type: ignore
            # t_start_time = Time(seconds=t_start.header.stamp.sec, nanoseconds=t_start.header.stamp.nanosec)

        t_start_mat, t_end_mat = matrix_from_transform(t_start), matrix_from_transform(t_end)

        points = self.laserscan_to_points(scan) if points is None else points
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

        # add the origin
        laser_position = t_interp_mats[int(len(t_interp_mats)/2)][:2, 3]
        points_deskewed = np.vstack((laser_position, points_deskewed, laser_position))

        return points_deskewed, start_time

    def deskew_laserscan_sample_tf(self, scan: LaserScan, steps: int = 10) -> Tuple[np.ndarray, Time]:
        """Deskew a ROS Laserscan by sample multiple transforms"""
        end_time = Time(seconds=scan.header.stamp.sec, nanoseconds=scan.header.stamp.nanosec)
        start_time = end_time - Duration(seconds=self.scan_time)  # type: ignore
        laser_frame = scan.header.frame_id

        ranges = np.array(scan.ranges)
        mask_invalid = (ranges < scan.range_min) | (ranges > scan.range_max)
        ranges[mask_invalid] = 0
        N = len(ranges)
        angles = scan.angle_min + np.arange(N) * scan.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points = (ranges * cos_sin_map).T

        indices = np.cumsum([N / steps] * (steps - 1)).astype(int)
        points_splitted = np.split(points, indices)
        points_splitted_transf = []
        for dt, points_part in zip(np.linspace(0, self.scan_time, steps + 1)[1:], points_splitted):
            eval_time = start_time + Duration(seconds=dt)  # type: ignore
            points_part_transf = self.transform_pointcloud(points_part, self.deskew_frame, laser_frame, eval_time)
            points_splitted_transf.append(points_part_transf)

        return np.vstack(points_splitted_transf)[~mask_invalid], start_time

    def transform_pointcloud(
        self, points: np.ndarray, target_frame: str, source_frame: str, time_stamp: Time
    ) -> np.ndarray:
        """Converts a pointcloud of 2D points (np.ndarray with shape = (N, 2)) with a ROS transform.
        Raises a TransformException when the transform is not available"""
        assert points.shape[1] == 2, "Should be 2D points with array shape (N, 2)"
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, time_stamp)
        points_4D = np.hstack((points, np.zeros((len(points), 1)), np.ones((len(points), 1))))
        t_mat = matrix_from_transform(t)
        return (t_mat @ points_4D.T)[:2].T

    def points_to_pointcloud_msg(self, points: np.ndarray, start_time: Time) -> PointCloud:
        """Convert a numpy array of points to a ROS PointCloud message"""
        header = Header(stamp=start_time.to_msg(), frame_id=self.deskew_frame)
        point32_list = [Point32(x=p[0], y=p[1]) for p in points.astype(np.float_)]
        return PointCloud(header=header, points=point32_list)
    
    def points_to_linestrip_msg(self, points: np.ndarray, namespace: str, marker_idx: int = 0) -> Marker:
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.deskew_frame)

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
            color=ColorRGBA(b=1., a=1.),
            ns=namespace,
        )
        return marker


def main(args=None):
    rclpy.init(args=args)

    save_topics_node = DeskewLidarNode()

    rclpy.spin(save_topics_node)
    save_topics_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
