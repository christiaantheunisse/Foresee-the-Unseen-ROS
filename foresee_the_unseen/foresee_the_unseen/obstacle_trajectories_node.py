import rclpy
from typing import List, Callable, Tuple, Type, Optional
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Duration
import time
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import copy

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory as TrajectoryCR
from commonroad.scenario.state import InitialState

from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg
from geometry_msgs.msg import Point, Vector3, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.obstacle_trajectories import (
    read_obstacle_configuration_yaml,
    generate_waypoints,
    to_ros_trajectory_msg,
    generate_velocity_profile,
    velocity_profile_to_state_list,
    get_ros_trajectory_from_commonroad_trajectory,
    get_ros_pose_from_commonroad_state,
)
from foresee_the_unseen.lib.helper_functions import polygons_from_road_xml, matrix_from_transform


class ObstacleTrajectoriesNode(Node):
    def __init__(self):
        super().__init__("obstacle_trajectories_node")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("planner_frame", "planner")
        self.declare_parameter("trajectory_topic", "trajectory")
        # self.declare_parameter("obstacle_config_yaml", "path/to/configuration.yaml")
        self.declare_parameter(
            "obstacle_config_yaml",
            "/home/christiaan/thesis/robot_ws/src/foresee_the_unseen/resource/obstacle_trajectories.yaml",
        )
        self.declare_parameter("do_visualize", False)
        self.declare_parameter("visualization_topic", "visualization/obstacle_trajectories")
        self.declare_parameter("startup_topic", "/goal_pose")

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value
        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.yaml_file = self.get_parameter("obstacle_config_yaml").get_parameter_value().string_value
        self.obstacle_config = read_obstacle_configuration_yaml(self.yaml_file)
        self.do_visualize = self.get_parameter("do_visualize").get_parameter_value().bool_value
        self.visualization_topic = self.get_parameter("visualization_topic").get_parameter_value().string_value
        self.startup_topic = self.get_parameter("startup_topic").get_parameter_value().string_value

        # Transformer listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        scenario, _ = CommonRoadFileReader(self.obstacle_config["road_structure_xml"]).open()
        self.road_polygons = polygons_from_road_xml(self.obstacle_config["road_structure_xml"])
        lanelet_network = scenario.lanelet_network

        if self.do_visualize:
            self.marker_array_publisher = self.create_publisher(MarkerArray, self.visualization_topic, 10)
            markers = self.get_road_structure_marker()
            colors = itertools.cycle(mcolors.TABLEAU_COLORS.values())

        self.trajectory_publishers: List[Publisher] = []
        self.trajectories_cr_delay: List[Tuple[TrajectoryCR, float]] = []
        self.trajectories_msg: Optional[List[TrajectoryMsg]] = None
        self.initialpose_publishers: List[Publisher] = []
        self.initialstates_commonroad: List[InitialState] = []
        for namespace, car_dict in self.obstacle_config["obstacle_cars"].items():
            waypoints, _, _ = generate_waypoints(
                lanelet_network=lanelet_network,
                initial_position=car_dict["start_pose"][:2],
                movement=car_dict["movement"],
                waypoint_distance=self.obstacle_config["waypoint_distance"],
                reference_velocity=car_dict["velocity"],
                acceleration_limit=car_dict["acceleration"],
                goal_positions_per_movement=self.obstacle_config["movements"],
            )
            velocity_profile = generate_velocity_profile(
                waypoints,
                car_dict["velocity"],
                car_dict["acceleration"],
                car_dict["acceleration"],
                self.obstacle_config["dt"],
            )
            trajectory_commonroad = velocity_profile_to_state_list(
                velocity_profile, waypoints, self.obstacle_config["dt"]
            )
            self.trajectories_cr_delay.append((trajectory_commonroad, car_dict["start_delay"]))
            
            publisher_traj = self.create_publisher(TrajectoryMsg, f"/{namespace}/{self.trajectory_topic}", 1)
            self.trajectory_publishers.append(publisher_traj)

            publisher_pose = self.create_publisher(PoseWithCovarianceStamped, f"/{namespace}/initialpose", 1)
            self.initialpose_publishers.append(publisher_pose)
            self.initialstates_commonroad.append(trajectory_commonroad.state_list[0])

            if self.do_visualize:
                markers += self.get_trajectory_marker(waypoints, namespace, next(colors))

        self.trajectory_publish_timer = self.create_timer(1, self.publish_trajectory_callback)
        self.trajectory_publish_timer.cancel()
        self.initialpose_publish_timer = self.create_timer(1, self.publish_initialpose_callback)

        # self.setup_timer_w_delay(namespace=namespace, delay=car_dict["start_delay"], msg=trajectory_msg)

        if self.do_visualize:
            self.create_timer(0.5, lambda: self.marker_array_publisher.publish(MarkerArray(markers=markers)))
            self.get_logger().info("trajectories visualized")

        self.create_subscription(PoseStamped, self.startup_topic, self.startup_callback, 1)
        # self.get_logger().info(
        #     str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.get_clock().now().seconds_nanoseconds()[0])))
        # )

    def startup_callback(self, msg: PoseStamped) -> None:
        """This function allows the robot to start moving."""
        if self.trajectories_msg is None:
            # Stop publishing the initial pose
            self.initialpose_publish_timer.cancel()

            self.trajectories_msg = []
            for trajectory_cr, delay in self.trajectories_cr_delay:
                trajectory_msg = get_ros_trajectory_from_commonroad_trajectory(
                    trajectory_cr,
                    self.get_clock().now() + Duration(seconds=delay),
                    self.obstacle_config["dt"],
                )
                self.trajectories_msg.append(trajectory_msg)

            self.trajectory_publish_timer.reset()

    def publish_trajectory_callback(self) -> None:
        """Publish the trajectories"""
        assert self.trajectories_msg is not None
        assert len(self.trajectory_publishers) == len(self.trajectories_msg), (
            "Ensure that the trajectories are converted to ROS Trajectory messages."
            + f" No. of publishers = {len(self.trajectory_publishers)}, "
            + f"No. of trajectory messages = {len(self.trajectories_msg)}"
        )
        for publisher, trajectory in zip(self.trajectory_publishers, self.trajectories_msg, strict=True):
            publisher.publish(trajectory)

    def publish_initialpose_callback(self) -> None:
        """Publish the initialpose for the other robots until the execution starts."""
        for publisher, state_planner in zip(self.initialpose_publishers, self.initialstates_commonroad):
            try:
                # self.get_logger().info(str(state_planner))
                # self.get_logger().info(str(publisher))
                transform = self.tf_buffer.lookup_transform("map", "planner", rclpy.time.Time())
                state_map = self.transform_state(state_planner, transform)
                # self.get_logger().info(str(state_map))
                pose_msg = get_ros_pose_from_commonroad_state(state_map)
                publisher.publish(pose_msg)
            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform `planner` to `map`: {ex}",
                    throttle_duration_sec=3,
                )

    @staticmethod
    def transform_state(state: InitialState, transform: TransformStamped) -> InitialState:
        """Converts a Commonroad state (2D position, orientation and scalar velocity) based on a ROS transform"""
        t_matrix = matrix_from_transform(transform)
        position = np.hstack((state.position, [0, 1]))  # 4D position
        state_tf = copy.deepcopy(state)
        state_tf.position = (t_matrix @ position)[:2]
        r_matrix = t_matrix[:2, :2]
        state_tf.orientation = np.arctan2(
            *np.flip(r_matrix @ np.array([np.cos(state_tf.orientation), np.sin(state_tf.orientation)]))
        )

        return state_tf
    
    # def setup_timer_w_delay(self, namespace: str, delay: float, msg: TrajectoryMsg) -> Callable[[], None]:
    #     # create the publisher
    #     publisher = self.create_publisher(TrajectoryMsg, f"/{namespace}/{self.trajectory_topic}", 1)

    #     def timer_callback() -> None:
    #         # publish the message only once
    #         self.get_logger().info(
    #             (
    #                 f"[timer_callback {namespace} ({delay=})] "
    #                 + time.strftime(
    #                     "%Y-%m-%d %H:%M:%S", time.localtime(self.get_clock().now().seconds_nanoseconds()[0])
    #                 )
    #             )
    #         )
    #         publisher.publish(msg)
    #         # self.get_logger().info(f"message published: {msg}")
    #         timer = getattr(self, f"timer_{namespace}")
    #         # timer.cancel()

    #     setattr(self, f"timer_{namespace}", self.create_timer(delay, timer_callback))

        return timer_callback

    def get_road_structure_marker(self) -> List[Marker]:
        """Get the markers that visualize the road boundaries."""
        markers = []
        for lanelet_id, points in self.road_polygons.items():
            point_type_list = []
            points = points
            for point in points:
                point_type_list.append(Point(x=point[0], y=point[1]))

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
            lanelet_marker = Marker(
                header=header,
                id=int(lanelet_id),
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                points=point_type_list,
                scale=Vector3(x=0.005),
                color=ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
                ns="road structure",
            )

            markers.append(lanelet_marker)
        return markers

    def get_trajectory_marker(self, waypoints: np.ndarray, namespace: str, color_hex: str) -> List[Marker]:
        """Get a marker that visualizes the trajectory"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="planner")

        color_hex = color_hex[1:] if color_hex[0] == "#" else color_hex
        color_rgb = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (0, 2, 4))

        point_list = []
        color_rgba_list = []
        # cmap = plt.get_cmap("cool")
        for x, y in waypoints:
            point_list.append(Point(x=float(x), y=float(y)))
            # r, g, b, a = cmap(len(point_list) / len(waypoints))
            r, g, b = color_rgb
            a = 0.2
            color_rgba_list.append(ColorRGBA(r=r, g=g, b=b, a=a))
        trajectory_marker = Marker(
            header=header,
            type=Marker.POINTS,
            action=Marker.MODIFY,
            id=0,
            points=point_list,
            colors=color_rgba_list,
            scale=Vector3(x=0.06, y=0.06),
            ns=namespace,
        )

        return [trajectory_marker]


def main(args=None):
    rclpy.init(args=args)

    obstacle_trajectories_node = ObstacleTrajectoriesNode()

    rclpy.spin(obstacle_trajectories_node)
    obstacle_trajectories_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
