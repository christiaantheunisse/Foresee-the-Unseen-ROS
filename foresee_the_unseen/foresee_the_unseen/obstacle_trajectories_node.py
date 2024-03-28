import rclpy
from typing import List, Callable, Tuple, Type
from rclpy.node import Node

from commonroad.common.file_reader import CommonRoadFileReader

from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg

from foresee_the_unseen.lib.obstacle_trajectories import (
    read_obstacle_configuration_yaml,
    generate_waypoints,
    to_ros_trajectory_msg,
)

import time

class ObstacleTrajectoriesNode(Node):
    def __init__(self):
        super().__init__("obstacle_trajectories_node")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("planner_frame", "planner")
        self.declare_parameter("trajectory_topic", "trajectory")
        self.declare_parameter("obstacle_config_yaml", "path/to/configuration.yaml")

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value
        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.yaml_file = self.get_parameter("obstacle_config_yaml").get_parameter_value().string_value
        self.obstacle_config = read_obstacle_configuration_yaml(self.yaml_file)

        scenario, _ = CommonRoadFileReader(self.obstacle_config["road_structure_xml"]).open()
        lanelet_network = scenario.lanelet_network

        for namespace, car_dict in self.obstacle_config["obstacle_cars"].items():
            waypoints, orientations, velocities = generate_waypoints(
                lanelet_network=lanelet_network,
                initial_position=car_dict["start_pose"][:2],
                movement=car_dict["movement"],
                waypoint_distance=self.obstacle_config["waypoint_distance"],
                reference_velocity=car_dict["velocity"],
                acceleration_limit=car_dict["acceleration"],
                goal_positions_per_movement=self.obstacle_config["movements"],
            )

            msg = to_ros_trajectory_msg(waypoints, orientations, velocities)
            self.setup_timer_w_delay(namespace=namespace, delay=car_dict["start_delay"], msg=msg)

        self.get_logger().info(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.get_clock().now().seconds_nanoseconds()[0]))))

    def setup_timer_w_delay(self, namespace: str, delay: float, msg: TrajectoryMsg) -> Callable[[], None]:
        # create the publisher
        publisher = self.create_publisher(TrajectoryMsg, f"/{namespace}/{self.trajectory_topic}", 1)

        def timer_callback() -> None:
            # publish the message only once
            self.get_logger().info((f"[timer_callback {namespace} ({delay=})] " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.get_clock().now().seconds_nanoseconds()[0]))))
            publisher.publish(msg)
            timer = getattr(self, f"timer_{namespace}")
            timer.cancel()

        setattr(self, f"timer_{namespace}", self.create_timer(delay, timer_callback))

        return timer_callback


def main(args=None):
    rclpy.init(args=args)

    obstacle_trajectories_node = ObstacleTrajectoriesNode()

    rclpy.spin(obstacle_trajectories_node)
    obstacle_trajectories_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
