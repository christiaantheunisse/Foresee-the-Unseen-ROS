import rclpy
import os
import numpy as np
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Int32
from sensor_msgs.msg import LaserScan, PointCloud

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


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

def matrix_from_transform(t):
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
            rot_matrix = rot_scipy.as_matrix()
            t_matrix = np.zeros((4,4))
            t_matrix[:3, :3] = rot_matrix
            t_matrix[:3, 3] = translation
            t_matrix[3, 3] = 1

            return t_matrix

class DeskewLidarNode(Node):
    def __init__(self):
        super().__init__("deskew_lidar")

        self.scan_topic = "scan"
        self.deskew_scan_topic = "deskewed_scan"
        self.deskew_frame = "odom"
        self.scan_frequency = 8
        self.scan_time = 1 / self.scan_frequency

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 5)
        self.deskew_scan_publisher = self.create_publisher(PointCloud, self.deskew_scan_topic, 5)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def scan_callback(self, msg: LaserScan) -> None:
        """The scan callback"""
        try:
            points = self.deskew_laserscan(msg)
            pointcloud_msg = self.points_to_pointcloud_msg(points)
            self.deskew_scan_publisher.publish(pointcloud_msg)
        except TransformException:
            pass
        
    def deskew_laserscan(self, scan: LaserScan, steps: int = 10) -> np.ndarray:
        """Deskew a ROS Laserscan"""
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

        indices = np.cumsum([N/steps] * (steps - 1)).astype(int)
        points_splitted = np.split(points, indices)
        points_splitted_transf = []
        for dt, points_part in zip(np.linspace(0, self.scan_time, steps+1)[1:], points_splitted):
            eval_time = start_time + Duration(seconds=dt)  # type: ignore
            points_part_transf = self.transform_pointcloud(points_part, self.deskew_frame, laser_frame, eval_time)
            points_splitted_transf.append(points_part_transf)

        return np.vstack(points_splitted_transf)

    def transform_pointcloud(self, points: np.ndarray, target_frame: str, source_frame: str, time_stamp: Time) -> np.ndarray:
        """Converts a pointcloud of 2D points (np.ndarray with shape = (N, 2)) with a ROS transform.
        Raises a TransformException when the transform is not available"""
        assert points.shape[1] == 2, "Should be 2D points with array shape (N, 2)"
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, time_stamp)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {source_frame} to {target_frame}: {ex}",
                throttle_duration_sec=3,
            )
            raise TransformException
        points_4D = np.hstack((points, np.zeros((len(points), 1)), np.ones((len(points), 1))))
        t_mat = matrix_from_transform(t)
        return (t_mat @ points_4D.T)[:2].T

    def points_to_pointcloud_msg(self, points: np.ndarray) -> PointCloud:
        """Convert a numpy array of points to a ROS PointCloud message"""
        raise NotImplementedError


def main(args=None):
    rclpy.init(args=args)

    save_topics_node = DeskewLidarNode()

    rclpy.spin(save_topics_node)
    save_topics_node.destroy_node()
    rclpy.shutdown()
