import rclpy
import os
import pickle
import yaml
import numpy as np
from typing import List, Callable, Union
from rclpy.node import Node
from dataclasses import dataclass
from ament_index_python import get_package_share_directory

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from racing_bot_interfaces.msg import ProjectedOccludedArea

from foresee_the_unseen.lib.helper_functions import create_log_directory


@dataclass
class TopicToStore:
    topic_name: str
    message_type: Callable[[], object]
    throttle_filter: int = 1  # store only one of the n messages
    queue_size: int = 15

    def __post_init__(self):
        self.topic_name_stripped = self.topic_name.replace('/', '__')


class LoggingNode(Node):
    """Logs the necessary topics for the experiments"""
    def __init__(self):
        super().__init__("logging_node")
        topics_to_store: List[TopicToStore] = []

        topics_to_store.extend(
            [
                TopicToStore(topic_name="/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car1/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car2/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car3/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car4/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car5/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car6/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/obstacle_car7/odometry/filtered", message_type=Odometry),
                TopicToStore(topic_name="/projected_occluded_area", message_type=ProjectedOccludedArea),
                TopicToStore(topic_name="/goal_pose", message_type=PoseStamped),
            ]
        )
        assert self.unique_check(topics_to_store), "All topic names should be unique"

        self.topic_names_to_search_for = ["*/odometry/filtered"]
        # hardcoded directory
        self.base_dir = "/home/christiaan/thesis/experiments_logging"
        try:
            loadpath = os.path.join(get_package_share_directory("foresee_the_unseen"), "resource", "commonroad_scenario.yaml")
            with open(loadpath, "r") as f:
                config = yaml.safe_load(f)
            experiment_name = config["experiment_name"]
        except (FileNotFoundError, KeyError) as ex:
            self.get_logger().error("Could not obtain experiment name: " + str(ex))
            experiment_name = None
            
        self.log_dir = create_log_directory(self.base_dir, experiment_name)

        callbacks = [self.create_callback(topic) for topic in topics_to_store]
        for topic, callback in zip(topics_to_store, callbacks):
            self.create_subscription(topic.message_type, topic.topic_name, callback, topic.queue_size)

        # store commonroad_scenarios.yaml and ros_params_scenario.yaml
        configuration_dir = os.path.join(self.log_dir, "configuration")
        os.mkdir(configuration_dir)
        for file in ["commonroad_scenario.yaml", "ros_params_scenario.yaml"]:
            loadpath = os.path.join(get_package_share_directory("foresee_the_unseen"), "resource", file)
            with open(loadpath, "r") as f:
                config = yaml.safe_load(f)
            savepath = os.path.join(os.path.join(self.log_dir, "configuration", file))
            with open(savepath, "w") as f:
                print(f)
                yaml.safe_dump(config, f)
    
    @staticmethod
    def unique_check(topics_to_store) -> bool:
        """Verifies if all the topic names and stripped names are unique"""
        names, stripped_names = np.array([(t.topic_name, t.topic_name_stripped) for t in topics_to_store]).T
        return len(np.unique(names)) == len(names) and len(np.unique(stripped_names)) == len(stripped_names)

    def setup_topic_to_store(self, topics_to_store: Union[TopicToStore, List[TopicToStore]]) -> None:
        topics_to_store = topics_to_store if isinstance(topics_to_store, list) else [topics_to_store]
        for topic in topics_to_store:
            callback = self.create_callback(topic)
            self.create_subscription(topic.message_type, topic.topic_name, callback, topic.queue_size)

    def create_callback(self, topic: TopicToStore) -> Callable[[Callable[[], object]], None]:
        """Creates the callback function for this topic."""
        setattr(self, str(topic.topic_name_stripped + "_counter"), 0)
        def callback(msg) -> None:
            counter = getattr(self, str(topic.topic_name_stripped + "_counter"))
            setattr(self, str(topic.topic_name_stripped + "_counter"), counter + 1)
            
            if counter % topic.throttle_filter == 0:
                filename = os.path.join(self.log_dir, f"{topic.topic_name_stripped} {counter}.pickle")
                with open(filename, "wb") as handle:
                    pickle.dump(msg, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return callback

def main(args=None):
    rclpy.init(args=args)

    logging_node = LoggingNode()

    rclpy.spin(logging_node)
    logging_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

