import rclpy
from rclpy.node import Node
import os
import yaml
import pickle
import copy
import numpy as np
from setuptools import find_packages
from ament_index_python import get_package_share_directory

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.geometry.shape import Rectangle, Circle

from shapely.geometry import Polygon as ShapelyPolygon

from datmo.msg import TrackArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3, Point, Point32, PolygonStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    matrices_from_cw_cvx_polygon,
)
from foresee_the_unseen.lib.planner import Planner
from foresee_the_unseen.lib.sensor import Sensor
from foresee_the_unseen.lib.occlusion_tracker import Occlusion_tracker
from foresee_the_unseen.lib.utilities import add_no_stop_zone
from foresee_the_unseen.lib.helper_functions import euler_from_quaternion, create_log_directory


RGBA = ["r", "g", "b", "a"]
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.0, 1.0, 1.0]
TRANSPARENT_BLUE = [0.0, 0.0, 1.0, 0.6]
BLACK = [0.0, 0.0, 0.0, 1.0]
TRANSPARENT_GREY = [0.5, 0.5, 0.5, 0.8]

XYZ = ["x", "y", "z"]


# TODO: Everything in this node should be implemented in the foresee_the_unseen node
class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        self.throttle_duration = 2 # s

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "laser")
        self.declare_parameter("planner_frame", "planner")

        self.declare_parameter("lidar_topic", "scan")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("datmo_topic", "datmo/box_kf")
        self.declare_parameter("filtered_lidar_topic", "scan/road_env")
        self.declare_parameter("fov_topic", "visualization/fov")
        self.declare_parameter("obstacles_topic", "visualization/obstacles")

        self.declare_parameter(
            "road_xml",
            os.path.join(get_package_share_directory("foresee_the_unseen"), "resource/road_structure_15.xml"),
            ParameterDescriptor(
                description="The file name of the .xml file in the resources folder describing the road structure"
            ),
        )
        self.declare_parameter(
            "foresee_the_unseen_yaml",
            os.path.join(get_package_share_directory("foresee_the_unseen"), "resource/commonroad_scenario.yaml"),
            ParameterDescriptor(
                description="The file name of the .yaml file in the resources folder describing the foresee-the-unseen setup"
            ),
        )
        self.declare_parameter("log_directory", descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))

        # self.declare_parameter(
        #     "road_translation", [0.0, 0.0], ParameterDescriptor(description="translation of the road")
        # )
        # self.declare_parameter(
        #     "road_rotation", 0.0, ParameterDescriptor(description="CCW rotation of the road structure")
        # )
        self.declare_parameter(
            "environment_boundary",
            [2.0, 3.0, 2.0, 0.0, -2.0, 0.0, -2.0, 3.0],
            ParameterDescriptor(
                description="Convex polygon describing part of the map frame visuable to the lidar of the ego_vehicle"
                "in the foresee-the-unseen algorithm. [x1, y1, x2, y2, ...]"
            ),
        )
        self.declare_parameter("ego_vehicle_size", [0.3, 0.18, 0.12])  # L x W x H [m]
        self.declare_parameter("ego_vehicle_offset", [0.0, 0.0, 0.06])  # L x W x H [m]
        self.declare_parameter("planner_frequency", 2.0)

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value

        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.datmo_topic = self.get_parameter("datmo_topic").get_parameter_value().string_value
        self.filtered_lidar_topic = self.get_parameter("filtered_lidar_topic").get_parameter_value().string_value
        self.fov_topic = self.get_parameter("fov_topic").get_parameter_value().string_value
        self.markersarray_topic = self.get_parameter("obstacles_topic").get_parameter_value().string_value

        self.road_xml = self.get_parameter("road_xml").get_parameter_value().string_value
        self.config_yaml = self.get_parameter("foresee_the_unseen_yaml").get_parameter_value().string_value
        self.log_root_directory = self.get_parameter("log_directory").get_parameter_value().string_value
        # self.road_translation = np.array(
        #     self.get_parameter("road_translation").get_parameter_value().double_array_value
        # )
        # self.road_rotation = self.get_parameter("road_rotation").get_parameter_value().double_value
        self.filter_polygon = np.array(
            self.get_parameter("environment_boundary").get_parameter_value().double_array_value
        ).reshape(-1, 2)
        self.ego_vehicle_size = self.get_parameter("ego_vehicle_size").get_parameter_value().double_array_value
        self.ego_vehicle_offset = self.get_parameter("ego_vehicle_offset").get_parameter_value().double_array_value
        self.frequency = self.get_parameter("planner_frequency").get_parameter_value().double_value

        # get road_structure
        # self.road_polygons = polygons_from_road_xml(self.road_xml, self.road_translation, self.road_rotation)
        self.get_logger().debug(f"road .xml file used: {self.road_xml}")
        self.road_polygons = polygons_from_road_xml(self.road_xml)

        # initialize commonroad scenario
        self.initialize_commonroad()

        # Timer for the visualization -> different timers for fov and other things
        self.create_timer(1 / self.frequency, self.visualize_callback)
        self.create_timer(1 / self.frequency, self.update_scenario)

        # Transformer listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers and publishers
        self.create_subscription(LaserScan, self.lidar_topic, self.laser_callback, 5)  # Laser scan
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)  # Ego vehicle pose
        self.create_subscription(TrackArray, self.datmo_topic, self.datmo_callback, 5)  # Ego vehicle pose
        self.filtered_laser_publisher = self.create_publisher(LaserScan, self.filtered_lidar_topic, 5)
        self.marker_array_publisher = self.create_publisher(MarkerArray, self.markersarray_topic, 10)
        # self.fov_publisher = self.create_publisher(PolygonStamped, self.fov_topic, 1) # FOV publisher

        self.laser_filter_matrices = matrices_from_cw_cvx_polygon(self.filter_polygon)

        # Foresee-the-Unseen variables
        self.detected_obstacles = None
        self.ego_vehicle_state = None
        self.trajectory = None
        self.planner_step = 0
        self.sensor_FOV = None

    def visualize_callback(self):
        markers = []
        markers += self.get_ego_vehicle_markers()
        markers += self.get_road_structure_markers()
        markers += self.get_flower_pots_markers()
        markers += self.get_filter_polygon()
        markers += self.get_fov_marker()

        self.marker_array_publisher.publish(MarkerArray(markers=markers))

    def get_ego_vehicle_markers(self):
        """Draw the ego vehicle bounding box"""

        ego_vehicle_bb = Marker()

        ego_vehicle_bb.header.frame_id = self.base_frame
        ego_vehicle_bb.id = 0
        ego_vehicle_bb.type = Marker.CUBE
        ego_vehicle_bb.action = Marker.ADD

        ego_vehicle_bb.pose.position = Point(**dict(zip(XYZ, self.ego_vehicle_offset)))
        ego_vehicle_bb.scale = Vector3(**dict(zip(XYZ, self.ego_vehicle_size)))
        ego_vehicle_bb.color = ColorRGBA(**dict(zip(RGBA, TRANSPARENT_BLUE)))
        ego_vehicle_bb.frame_locked = True
        ego_vehicle_bb.ns = "ego_vehicle"

        return [ego_vehicle_bb]

    def get_road_structure_markers(self):
        markers_list = []
        for lanelet_id, points in self.road_polygons.items():
            point_type_list = []
            points = points
            for point in points:
                point_type_list.append(Point(**dict(zip(XYZ, point))))

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
            lanelet_marker = Marker(
                header=header,
                id=int(lanelet_id),
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                points=point_type_list,
                scale=Vector3(x=0.005),
                color=ColorRGBA(**dict(zip(RGBA, BLACK))),
                ns="road structure",
            )

            markers_list.append(lanelet_marker)

        return markers_list

    def get_flower_pots_markers(self):
        left_pot, right_pot = Marker(), Marker()
        size = np.array([0.4, 0.4, 1], dtype=np.float_)
        pot_ids = [1, 2]
        for id_pot, pot in zip(pot_ids, [left_pot, right_pot]):
            pot.header.frame_id = self.map_frame
            pot.id = id_pot
            pot.type = Marker.CUBE
            pot.action = Marker.ADD
            pot.scale = Vector3(**dict(zip(XYZ, size)))
            pot.color = ColorRGBA(**dict(zip(RGBA, TRANSPARENT_GREY)))
            pot.ns = "flower pot"

        left_position = (np.array([1, 0.525, 0]) + [0, size[1] / 2, size[2] / 2]).astype(np.float_)
        right_position = (np.array([1, -0.525, 0]) + [0, -size[1] / 2, size[2] / 2]).astype(np.float_)
        left_pot.pose.position = Point(**dict(zip(XYZ, left_position)))
        right_pot.pose.position = Point(**dict(zip(XYZ, right_position)))

        return [left_pot, right_pot]

    def laser_callback(self, msg):
        self.filter_pointcloud(msg)
        self.FOV_from_laser(msg)

    def filter_pointcloud(self, scan_msg: LaserScan):
        """Filter points from the pointcloud from the lidar to reduce the computation of the `datmo` package."""
        # TODO: transform the bounds instead of the laserscan itself

        # Convert the ranges to a pointcloud
        ranges = np.array(scan_msg.ranges)

        N = len(ranges)
        angles = scan_msg.angle_min + np.arange(N) * scan_msg.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points_laser = ranges * cos_sin_map  # 2D points
        points_laser = np.nan_to_num(points_laser, nan=1e99, posinf=1e99, neginf=-1e99)

        points_laser = np.vstack((points_laser, np.zeros((1, len(angles)))))
        # points_laser = points_laser[((ranges > scan_msg.range_min) & (ranges < scan_msg.range_max))] # apply the bounds
        points_laser_4D = np.vstack((points_laser, np.full((1, points_laser.shape[1]), 1)))

        # Convert to map frame and filter base on ENVIRONMENT
        try:
            t_map_laser = self.tf_buffer.lookup_transform(self.map_frame, self.lidar_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {self.lidar_frame} to {self.map_frame}: {ex}", throttle_duration_sec=self.throttle_duration)
            return None
        t_mat_map_laser = matrix_from_transform(t_map_laser)
        points_map = (t_mat_map_laser @ points_laser_4D)[:2]

        # filter the points based on a polygon
        A, B = self.laser_filter_matrices
        mask_polygon = np.all(A @ points_map <= np.repeat(B.reshape(-1, 1), points_map.shape[1], axis=1), axis=0)

        # Convert to the base frame and filter based on FOV
        try:
            t_base_laser = self.tf_buffer.lookup_transform(self.base_frame, self.lidar_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {self.base_frame} to {self.map_frame}: {ex}", throttle_duration_sec=self.throttle_duration)
            return None
        t_mat_base_laser = matrix_from_transform(t_base_laser)
        points_base = (t_mat_base_laser @ points_laser_4D)[:2]
        # filter the points based on a circle -> distance from origin
        dist = np.linalg.norm(points_base, axis=0)
        mask_fov = dist <= 2

        mask = mask_polygon & mask_fov
        new_msg = scan_msg
        # new_msg.header.frame_id = self.map_frame
        ranges = np.array(scan_msg.ranges).astype(np.float32)
        ranges[~mask] = np.inf  # set filtered out values to infinite
        new_msg.ranges = ranges

        self.filtered_laser_publisher.publish(new_msg)

    def get_filter_polygon(self):
        point_list = []
        for point in [*self.filter_polygon, self.filter_polygon[0]]:
            point_list.append(Point(**dict(zip(XYZ, np.array(point, dtype=np.float_)))))

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)
        filter_marker = Marker(
            header=header,
            id=0,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            points=point_list,
            scale=Vector3(x=0.01),
            color=ColorRGBA(**dict(zip(RGBA, BLUE))),
            ns="experiment env",
        )

        return [filter_marker]

    def publish_fov(self):
        polygon = PolygonStamped()
        polygon.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.base_frame)

        angles = np.linspace(0, 2 * np.pi, 20)
        points = np.array([np.cos(angles), np.sin(angles)], dtype=np.float_).T * 2
        point32_list = []
        for point in [*points, points[0]]:
            point32_list.append(Point32(**dict(zip(XYZ, point))))

        polygon.polygon.points = point32_list
        self.fov_publisher.publish(polygon)

    def get_fov_marker(self):
        angles = np.linspace(0, 2 * np.pi, 20)
        points = np.array([np.cos(angles), np.sin(angles)], dtype=np.float_).T * 2
        point_list = []
        for point in [*points, points[0]]:
            point_list.append(Point(**dict(zip(XYZ, np.array(point, dtype=np.float_)))))

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.base_frame)
        fov_marker = Marker(
            header=header,
            id=0,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            points=point_list,
            scale=Vector3(x=0.01),
            color=ColorRGBA(**dict(zip(RGBA, BLUE))),
            frame_locked=True,
            ns="fov",
        )

        return [fov_marker]

    # commonroad functions
    # ==================================================================================================================
    def datmo_callback(self, msg: TrackArray):
        """Converts DATMO detections to Commonroad obstacles"""
        # TODO: Implement
        # TODO: convert to planner frame
        self.detected_obstacles = []
        self.get_logger().debug(f"Obstacle detections updated: no. of detections = {len(self.detected_obstacles)}")

    def odom_callback(self, msg: Odometry):
        """Converts the odometry data to a Commonroad state."""
        # TODO: convert odom message to the planner frame
        source_frame = msg.header.frame_id
        try:
            t_planner_odom = self.tf_buffer.lookup_transform(self.planner_frame, source_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {source_frame} to {self.planner_frame}: {ex}", throttle_duration_sec=self.throttle_duration)
            return None

        state_untransformed = self.state_from_odom(msg, self.planner_step)
        self.ego_vehicle_state = self.transform_state(copy.deepcopy(state_untransformed), t_planner_odom)
        self.get_logger().debug(f"Ego vehicle state updated: state = {self.ego_vehicle_state}")

    def initialize_commonroad(self):
        with open(self.config_yaml) as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)
        
        # Make directory for logging
        self.log_dir = create_log_directory(self.log_root_directory)
        
        # Save the settings in the log file
        settings_dict = self.configuration
        settings_dict['road_xml'] = self.road_xml
        settings_dict['frequency'] = self.frequency

        filename = os.path.join(self.log_dir, "settings.pickle")
        with open(filename, 'wb') as handle:
            pickle.dump(settings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.scenario, _ = CommonRoadFileReader(self.road_xml).open()

        # ego_vehicle_initial_state = InitialState(
        #     position=np.array([self.configuration.get("initial_state_x"), self.configuration.get("initial_state_y")]),
        #     orientation=self.configuration.get("initial_state_orientation"),
        #     velocity=self.configuration.get("initial_state_velocity"),
        #     time_step=0,
        # )
        # ego_vehicle_initial_state = InitialState()
        ego_shape = Rectangle(self.configuration.get("vehicle_length"), self.configuration.get("vehicle_width"))
        ego_vehicle_initial_state = InitialState(position=np.array([0, 0]), orientation=0, velocity=0, time_step=0)
        self.ego_vehicle = DynamicObstacle(
            self.scenario.generate_object_id(), ObstacleType.CAR, ego_shape, ego_vehicle_initial_state
        )

        self.sensor = Sensor(
            self.ego_vehicle.initial_state.position,
            field_of_view=self.configuration.get("field_of_view_degrees") * 2 * np.pi / 360,
            min_resolution=self.configuration.get("min_resolution"),
            view_range=self.configuration.get("view_range"),
        )

        self.occ_track = Occlusion_tracker(
            self.scenario,
            min_vel=self.configuration.get("min_velocity"),
            max_vel=self.configuration.get("max_velocity"),
            min_shadow_area=self.configuration.get("min_shadow_area"),
            prediction_horizon=self.configuration.get("prediction_horizon"),
            tracking_enabled=self.configuration.get("tracking_enabled"),
        )

        self.planner = Planner(
            self.ego_vehicle.initial_state,
            vehicle_shape=self.ego_vehicle.obstacle_shape,
            goal_point=[self.configuration.get("goal_point_x"), self.configuration.get("goal_point_y")],
            reference_speed=self.configuration.get("reference_speed"),
            max_acceleration=self.configuration.get("max_acceleration"),
            max_deceleration=self.configuration.get("max_deceleration"),
            time_horizon=self.configuration.get("planning_horizon"),
        )
        self.get_logger().info("commonroad scenario initialized")

    @staticmethod
    def update_ego_vehicle(ego_vehicle: DynamicObstacle, state: State):
        """Update the ego vehicle based on the state and not based on the predicted trajectory."""
        # TODO: maybe store prediction separately
        if ego_vehicle.prediction is not None:
            trajectory = Trajectory(1 + state.time_step, ego_vehicle.prediction.trajectory.state_list[1:])
            trajectory_prediction = TrajectoryPrediction(trajectory, ego_vehicle.obstacle_shape)
        else:
            trajectory_prediction = None

        return DynamicObstacle(
            obstacle_id=ego_vehicle.obstacle_id,
            obstacle_type=ego_vehicle.obstacle_type,
            obstacle_shape=ego_vehicle.obstacle_shape,
            initial_state=state,
            prediction=trajectory_prediction,
        )

    @staticmethod
    def state_from_odom(odom: Odometry, time_step: int = 0) -> State:
        """Convert a ROS odometry message to a commonroad state."""
        position = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
        ]
        quaternion = [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ]
        _, _, yaw = euler_from_quaternion(*quaternion)
        velocity = [
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
        ]
        velocity_along_heading = np.array([np.cos(yaw), np.sin(yaw)]) @ velocity
        return InitialState(
            position=np.array(position),
            orientation=yaw,  # along x is 0, ccw is positive
            velocity=velocity_along_heading,
            time_step=time_step,
        )
    
    # @staticmethod
    def transform_state(self, state: State, transform: TransformStamped) -> State:
        """Converts a Commonroad state (2D position, orientation and scalar velocity) based on a ROS transform"""
        t_matrix = matrix_from_transform(transform)
        position = np.hstack((state.position, [0, 1])) # 4D position
        state.position = (t_matrix @ position)[:2]
        r_matrix = t_matrix[:2, :2]
        state.orientation = np.arctan2(*np.flip(r_matrix @ np.array([np.cos(state.orientation), np.sin(state.orientation)])))

        return state
    
    def FOV_from_laser(self, scan_msg: LaserScan):
        """Determines the FOV based on the LaserScan message"""
        max_range = 2
        ranges = np.array(scan_msg.ranges)
        ranges[ranges > max_range] = max_range
        N = len(ranges)
        angles = scan_msg.angle_min + np.arange(N) * scan_msg.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points_laser = (ranges * cos_sin_map).T  # 2D points
        # points_laser = np.nan_to_num(points_laser, nan=1e99, posinf=1e99, neginf=-1e99)
        self.sensor_FOV = ShapelyPolygon(points_laser) if len(points_laser) >= 3 else None
        

    def update_scenario(self):
        """Gets called every n seconds"""
        # TODO: keep track of the time_step -- verify that it works
        if self.detected_obstacles is None or self.ego_vehicle_state is None:
            if self.detected_obstacles is None:
                self.get_logger().warn("No detected obstacles available")
            if self.ego_vehicle_state is None:
                self.get_logger().warn("No ego vehicle state available")
            return

        self.get_logger().debug("Starting the scenario update")
        percieved_scenario = copy.deepcopy(self.scenario)
        # update ego_vehicle state
        self.ego_vehicle = self.update_ego_vehicle(self.ego_vehicle, self.ego_vehicle_state)

        percieved_scenario.add_objects(self.detected_obstacles)

        # Update the sensor and get the sensor view and the list of observed obstacles
        # self.sensor.update(self.ego_vehicle.initial_state)
        # sensor_view = self.sensor.get_sensor_view(percieved_scenario)

        if self.sensor_FOV is not None:
            sensor_view = self.sensor_FOV
        else:
            self.get_logger().warn("No detected obstacles available", throttle_duration_sec=self.throttle_duration)
            return

        # Update the tracker with the new sensor view and get the prediction for the shadows
        self.occ_track.update(sensor_view, self.ego_vehicle.initial_state.time_step)
        shadow_obstacles = self.occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        add_no_stop_zone(
            percieved_scenario,
            self.planner_step + self.configuration.get("planning_horizon"),
            self.configuration.get("safety_margin"),
        )  # should not be necessary in every timestep
        
        
        self.planner.update(self.ego_vehicle.initial_state) # FIXME: should not be possible to remove all waypoints
        # FIXME: Throws an attribute error if the ego_vehicle or goal_position is not on the road
        collision_free_trajectory = self.planner.plan(percieved_scenario)
        if collision_free_trajectory:
            self.get_logger().info("new trajectory found")
            # self.ego_vehicle.prediction = collision_free_trajectory
            self.trajectory = collision_free_trajectory

        # TODO: publish trajectory

        # TODO: visualize trajectory

        # TODO: visualize occupancies

        percieved_scenario.add_objects(self.ego_vehicle)
        log_dict = {
            "ego_vehicle": self.ego_vehicle,
            "scenario": percieved_scenario,
            "sensor_view": sensor_view
        }
        filename = os.path.join(self.log_dir, f"step {self.planner_step}.pickle")
        with open(filename, 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO: RVIZ visualization

        self.planner_step += 1
        self.get_logger().info(f"Scenario updated: planner step = {self.planner_step}")


def main(args=None):
    rclpy.init(args=args)

    visualization_node = VisualizationNode()

    rclpy.spin(visualization_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    visualization_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
