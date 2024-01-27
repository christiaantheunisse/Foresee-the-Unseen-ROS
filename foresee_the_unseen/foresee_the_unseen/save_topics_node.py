import rclpy
import os
import numpy as np
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Int32
import math

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

class SaveTopicsNode(Node):
    def __init__(self):
        super().__init__("save_poses_node")
        self.encoder_left = {
            "topic_name": "left_wheel",
            "type": Int32,
            "queue_size": 10,
            "data_extract_func": self.Int32_data_extract_func,
        }
        self.encoder_right = {
            "topic_name": "right_wheel",
            "type": Int32,
            "queue_size": 10,
            "data_extract_func": self.Int32_data_extract_func,
        }
        self.slam_pose = {
            "topic_name": "pose",
            "type": PoseWithCovarianceStamped,
            "queue_size": 1,
            "data_extract_func": self.PoseWithCovarianceStamped_data_extract_func,
        }
        self.odometry = {
            "topic_name": "odom",
            "type": Odometry,
            "queue_size": 2,
            "data_extract_func": self.PoseWithCovarianceStamped_data_extract_func,
        }
        self.odometry_transformed = {
            "topic_name": "odom",
            "file_name": "odom_transformed",
            "type": Odometry,
            "queue_size": 2,
            "data_extract_func": self.odom_pose_in_map_frame_extract_func,
        }
        self.topics_to_save = [
            self.encoder_left,
            self.encoder_right,
            self.slam_pose,
            self.odometry,
            self.odometry_transformed,
        ]
        for topic in self.topics_to_save:
            topic["data"] = []
            topic["subscription"] = self.create_subscription(
                topic["type"],
                topic["topic_name"],
                self.create_callback(topic),
                topic["queue_size"],
            )

        self.timer = self.create_timer(5, self.timer_callback)
        self.dir_to_save = "/home/christiaan/thesis/npy_files"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def timer_callback(self):
        self.get_logger().info("Saving data to file...")
        for topic in self.topics_to_save:
            if topic.get("updated", True) or True:
                array = np.array(topic["data"])
                filename = topic.get("file_name", topic["topic_name"])
                np.save(os.path.join(self.dir_to_save, filename), array)
                topic["updated"] = False

    @staticmethod
    def Int32_data_extract_func(msg):
        return msg.data  # integer

    @staticmethod
    def PoseWithCovarianceStamped_data_extract_func(msg):
        position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        euler_angles = euler_from_quaternion(*quaternion)
        data = np.hstack((position, euler_angles))
        return data  # x, y, z, yaw, pitch, roll

    @staticmethod
    def Odometry_data_extract_func(msg):
        return SaveTopicsNode.PoseWithCovarianceStamped_data_extract_func(msg)

    def odom_pose_in_map_frame_extract_func(self, msg):
        to_frame_rel = "map"
        from_frame_rel = "odom"
        try:
            t = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {from_frame_rel} to {to_frame_rel}: {ex}")
            return None

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
        trans_matrix = np.zeros((4,4))
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = translation
        trans_matrix[3, 3] = 1

        # transform the position in the odom frame to the map frame
        position_odom_4D = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            1,
        ]
        position_map_4D = trans_matrix @ position_odom_4D
        return position_map_4D[:3]


    def create_callback(self, topic):
        def callback(msg):
            topic["updated"] = True
            data = topic["data_extract_func"](msg)
            if data is not None:
                topic["data"].append(data)

        return callback


def main(args=None):
    rclpy.init(args=args)

    save_topics_node = SaveTopicsNode()

    rclpy.spin(save_topics_node)
    save_topics_node.destroy_node()
    rclpy.shutdown()
