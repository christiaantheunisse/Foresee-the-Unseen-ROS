import rclpy
import os
import math
import pickle
import numpy as np
from typing import List, Callable
from rclpy.node import Node
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from racing_bot_interfaces.msg import Trajectory, ProjectedOccludedArea

from foresee_the_unseen.lib.helper_functions import create_log_directory


@dataclass
class TopicToStore:
    topic_name: str
    message_type: Callable[[], object]
    throttle_filter: int = 1  # store only one of the n messages
    queue_size: int = 15

    def __post_init__(self):
        self.topic_name_stripped = self.topic_name.replace('/', '__')


class StoreTopicsNode(Node):
    def __init__(self):
        super().__init__("topics_to_disk_node")
        topics_to_store: List[TopicToStore] = []

        self.get_topic_names_and_types()
        topics_to_store.extend(
            [
                # TopicToStore(topic_name="/scan", message_type=LaserScan),
                # TopicToStore(topic_name="/scan/road_env", message_type=LaserScan),
                # TopicToStore(topic_name="/imu_data", message_type=Imu),
                # TopicToStore(topic_name="/odometry/velocity_ekf", message_type=Odometry),
                # TopicToStore(topic_name="/odometry/wheel_encoders", message_type=Odometry),
                # TopicToStore(topic_name="/slam_pose", message_type=PoseWithCovarianceStamped),
                # TopicToStore(topic_name="/trajectory", message_type=Trajectory),

                # Logging for the experiments
                # TopicToStore(topic_name="/odometry/filtered", message_type=Odometry),
                # TopicToStore(topic_name="/projected_occluded_area", message_type=ProjectedOccludedArea),
                TopicToStore(topic_name="/goal_pose", message_type=PoseStamped),
                TopicToStore(topic_name="/obstacle_car1/trajectory", message_type=Trajectory),
                TopicToStore(topic_name="/obstacle_car2/trajectory", message_type=Trajectory),
                TopicToStore(topic_name="/obstacle_car1/odometry/filtered", message_type=Odometry),
                # TopicToStore(topic_name="/odometry/filtered", message_type=Odometry),
            ]
        )
        assert self.unique_check(topics_to_store), "All topic names should be unique"

        # hardcoded directory
        self.base_dir = "/home/christiaan/thesis/topic_store_files"
        self.log_dir = create_log_directory(self.base_dir)

        callbacks = [self.create_callback(topic) for topic in topics_to_store]
        for topic, callback in zip(topics_to_store, callbacks):
            self.create_subscription(topic.message_type, topic.topic_name, callback, topic.queue_size)

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

def main(args=None):
    rclpy.init(args=args)

    store_topics_node = StoreTopicsNode()

    rclpy.spin(store_topics_node)
    store_topics_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

