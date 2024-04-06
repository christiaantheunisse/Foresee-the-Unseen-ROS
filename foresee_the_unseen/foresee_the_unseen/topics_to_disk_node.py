import rclpy
import os
import math
import pickle
import numpy as np
from typing import List, Callable, Tuple, Type
from rclpy.node import Node
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import LaserScan
from racing_bot_interfaces.msg import EncoderValues, Trajectory
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import create_log_directory


@dataclass
class TopicToStore:
    topic_name: str
    message_type: Type
    throttle_filter: int = 1  # store only one of the n messages
    queue_size: int = 15

    def __post_init__(self):
        self.topic_name_stripped = self.topic_name.replace("/", "__")


class StoreTopicsNode(Node):
    def __init__(self):
        super().__init__("topics_to_disk_node")
        topics_to_store: List[TopicToStore] = []

        topics_to_store.extend(
            [
                TopicToStore(topic_name="/scan", message_type=LaserScan),
                # TopicToStore(topic_name="/scan/road_env", message_type=LaserScan),
                # TopicToStore(topic_name="/wheel_encoders", message_type=EncoderValues),
                # TopicToStore(topic_name="/odom", message_type=Odometry),
                # TopicToStore(topic_name="/odometry/filtered", message_type=Odometry),
                # TopicToStore(topic_name="/slam_pose", message_type=PoseWithCovarianceStamped),
                # TopicToStore(topic_name="/trajectory", message_type=Trajectory),
                # TopicToStore(topic_name="/cmd_motor", message_type=Int16MultiArray),
            ]
        )
        assert self.unique_check(topics_to_store), "All topic names should be unique"

        transforms_to_store: List[Tuple[str, str, float]] = []

        # transforms_to_store.extend([("map", "odom", 20)])
        # transforms_to_store.extend([("map", "planner", 20)])

        # hardcoded directory
        self.base_dir = "/home/christiaan/thesis/topic_store_files"
        self.log_dir = create_log_directory(self.base_dir)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        callbacks = [self.create_callback(topic) for topic in topics_to_store]
        for topic, callback in zip(topics_to_store, callbacks):
            self.create_subscription(topic.message_type, topic.topic_name, callback, topic.queue_size)

        for source_frame, target_frame, rate in transforms_to_store:
            self.create_transform_timer(source_frame, target_frame, rate)

    @staticmethod
    def unique_check(topics_to_store) -> bool:
        """Verifies if all the topic names and stripped names are unique"""
        names, stripped_names = np.array([(t.topic_name, t.topic_name_stripped) for t in topics_to_store]).T
        return len(np.unique(names)) == len(names) and len(np.unique(stripped_names)) == len(stripped_names)

    def create_callback(self, topic: TopicToStore) -> Callable[[Callable[[], object]], None]:
        """Creates the callback function for this topic."""
        setattr(self, str(topic.topic_name_stripped + "_counter"), 0)

        def callback(msg: topic.message_type) -> None:
            counter = getattr(self, str(topic.topic_name_stripped + "_counter"))
            setattr(self, str(topic.topic_name_stripped + "_counter"), counter + 1)

            if counter % topic.throttle_filter == 0:
                filename = os.path.join(self.log_dir, f"{topic.topic_name_stripped} {counter}.pickle")
                with open(filename, "wb") as handle:
                    pickle.dump(msg, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return callback
    
    def create_transform_timer(self, source_frame: str, target_frame: str, frequency: float) -> None:
        """ Creates a callback function for a timer to save a transform. """
        name = "transform_" + source_frame + "_" + target_frame
        setattr(self, str(name + "_counter"), 0)

        def callback() -> None:
            counter = getattr(self, str(name + "_counter"))
            setattr(self, str(name + "_counter"), counter + 1)
            
            try:
                transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {source_frame} to {target_frame}: {ex}",
                    throttle_duration_sec=3,
                )
                return None
            
            filename = os.path.join(self.log_dir, f"{name} {counter}.pickle")
            with open(filename, "wb") as handle:
                    pickle.dump(transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.create_timer(1/frequency, callback)


def main(args=None):
    rclpy.init(args=args)

    store_topics_node = StoreTopicsNode()

    rclpy.spin(store_topics_node)
    store_topics_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
