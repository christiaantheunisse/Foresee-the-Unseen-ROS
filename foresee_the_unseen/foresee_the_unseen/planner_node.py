import rclpy
from rclpy.node import Node
import os
import yaml
import time
import numpy as np
import matplotlib as mpl
from setuptools import find_packages
from ament_index_python import get_package_share_directory
from typing import Optional, List, Union, Tuple, Dict, TypedDict

from commonroad.scenario.state import InitialState, State
from commonroad.scenario.obstacle import Obstacle, ObstacleType, DynamicObstacle
from commonroad.geometry.shape import Rectangle

from shapely.geometry import Polygon as ShapelyPolygon

from datmo.msg import TrackArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3, Point, Point32, PolygonStamped, TransformStamped, Pose
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import ParameterDescriptor

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    matrices_from_cw_cvx_polygon,
    euler_from_quaternion,
)
from foresee_the_unseen.lib.foresee_the_unseen import ForeseeTheUnseen, NoUpdatePossible
from foresee_the_unseen.lib.triangulate import triangulate, remove_redudant_vertices_polygon


RGBA = ["r", "g", "b", "a"]
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.0, 1.0, 1.0]
TRANSPARENT_BLUE = [0.0, 0.0, 1.0, 0.6]
BLACK = [0.0, 0.0, 0.0, 1.0]
TRANSPARENT_GREY = [0.5, 0.5, 0.5, 0.8]

Color = TypedDict("Color", {"r": float, "g": float, "r": float, "a": float})
RED: Color = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
GREEN: Color = {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0}
BLUE: Color = {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
BLACK: Color = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
TRANSPARENT_BLUE: Color = {"r": 0.0, "g": 0.0, "b": 1.0, "a": 0.6}
TRANSPARENT_GREY: Color = {"r": 0.5, "g": 0.5, "b": 0.5, "a": 0.8}

SHADOW_OBS_COLOR: Color = {"r": 100.0 / 255.0, "g": 0.0, "b": 0.0, "a": 1.0}
SHADOW_PRED_COLOR: Color = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}

XYZ = ["x", "y", "z"]


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner_node")
        self.throttle_duration = 2  # s

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
        self.declare_parameter("log_directory", "none")

        self.declare_parameter(
            "environment_boundary",
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ParameterDescriptor(
                description="Convex polygon describing part of the map frame visuable to the lidar of the ego_vehicle"
                "in the foresee-the-unseen algorithm. [x1, y1, x2, y2, ...]"
            ),
        )
        self.declare_parameter("ego_vehicle_size", [0.3, 0.18, 0.12])  # L x W x H [m]
        self.declare_parameter("ego_vehicle_offset", [0.0, 0.0, 0.06])  # L x W x H [m]
        self.declare_parameter("planner_frequency", 2.0)

        self.declare_parameter("use_triangulation", False)  # for the visualization of the polygons
        self.declare_parameter("num_pred_to_visualize", 5)
        self.declare_parameter("do_visualize", True)

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
        self.filter_polygon = np.array(
            self.get_parameter("environment_boundary").get_parameter_value().double_array_value
        ).reshape(-1, 2)
        self.ego_vehicle_size = self.get_parameter("ego_vehicle_size").get_parameter_value().double_array_value
        self.ego_vehicle_offset = self.get_parameter("ego_vehicle_offset").get_parameter_value().double_array_value
        self.frequency = self.get_parameter("planner_frequency").get_parameter_value().double_value

        self.use_triangulation = self.get_parameter("use_triangulation").get_parameter_value().bool_value
        self.predictions_to_visualize = self.get_parameter("num_pred_to_visualize").get_parameter_value().integer_value
        self.do_visualize = self.get_parameter("do_visualize").get_parameter_value().bool_value

        # get road_structure and configuration
        self.get_logger().info(f"road .xml file used: {self.road_xml}")
        self.road_polygons = polygons_from_road_xml(self.road_xml)
        with open(self.config_yaml) as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)

        # Timer for the visualization -> different timers for fov and other things
        # self.create_timer(1 / self.frequency, self.visualize_callback)
        self.create_timer(1 / self.frequency, self.plan)

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

        self.foresee_the_unseen_planner = ForeseeTheUnseen(
            config_yaml=self.config_yaml,
            road_xml=self.road_xml,
            frequency=self.frequency,
            logger=self.get_logger(),
            log_dir=self.log_root_directory if self.log_root_directory != "none" else None,
        )

    # CALLBACKS
    def plan(self):
        """Calls the Foresee The Unseen Planner and updates the visualization"""
        start_time = time.time()
        try:
            perceived_scenario, sensor_view = self.foresee_the_unseen_planner.update_scenario()
            shadow_obstacles = [o for o in perceived_scenario.obstacles if o.obstacle_type == ObstacleType.UNKNOWN]
        except NoUpdatePossible:
            self.get_logger().warn("No planner update step possible")
            shadow_obstacles, sensor_view = None, None

        # TODO: Visualize shadow obstacles
        # TODO: Visualize occupancies of shadow obstacles
        # TODO: Visualize goal point

        start_viz_time = time.time()
        if self.do_visualize:
            self.visualization_callback(shadow_obstacles, sensor_view)
        end_time = time.time()
        tot_time = end_time - start_time
        viz_time = end_time - start_viz_time
        if tot_time > (1 / self.frequency):
            self.get_logger().error(
                f"Planner update took too long: execution time = {tot_time * 1000:.0f} ms "
                + f"(visualization = {viz_time * 1000:.0f} ms; maximum time {1/self.frequency*1000:.0f} ms "
                + f"({self.frequency} Hz)",
            )
        else:
            pass
            # self.get_logger().info(
            #     f"Planner update: execution time = {tot_time * 1000:.0f} ms "
            #     + f"(visualization = {viz_time * 1000:.0f} ms; maximum time {1/self.frequency*1000:.0f} ms "
            #     + f"({self.frequency} Hz)",
            # )

    def visualization_callback(
        self,
        shadow_obstacles: Optional[List[Obstacle]] = None,
        sensor_view: Optional[ShapelyPolygon] = None,
    ):
        # TODO: Only remove markers that need to and only add markers that need to be added every time step
        markers = []
        markers.append(Marker(action=Marker.DELETEALL)) # remove all markers
        markers += self.get_ego_vehicle_marker()
        markers += self.get_road_structure_marker()
        markers += self.get_filter_polygon_marker()
        markers += self.get_goal_marker()

        markers += self.get_fov_marker(sensor_view)
        markers += self.get_trajectory_marker()
        markers += self.get_shadow_marker(shadow_obstacles)
        self.marker_array_publisher.publish(MarkerArray(markers=markers))

    def laser_callback(self, msg):
        self.filter_pointcloud(msg)
        self.FOV_from_laser(msg)

    def datmo_callback(self, msg: TrackArray):
        """Converts DATMO detections to Commonroad obstacles"""
        if msg.tracks:  # length not zero
            datmo_frame = msg.tracks[0].odom.header.frame_id
            try:
                t_planner_datmo = self.tf_buffer.lookup_transform(self.planner_frame, datmo_frame, rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {datmo_frame} to {self.planner_frame}: {ex}",
                    throttle_duration_sec=self.throttle_duration,
                )
                return None
        detected_obstacles = []
        for track in msg.tracks:
            detected_obstacles.append(
                DynamicObstacle(
                    obstacle_id=0,  # Should be mutated
                    obstacle_type=ObstacleType.CAR,
                    obstacle_shape=Rectangle(track.length, track.width),
                    initial_state=self.transform_state(self.state_from_odom(track.odom), t_planner_datmo),
                )
            )

        self.foresee_the_unseen_planner.update_obstacles(detected_obstacles)

    def odom_callback(self, msg: Odometry):
        """Converts the odometry data to a Commonroad state and updates the Foresee the Unseen object."""
        odom_frame = msg.header.frame_id
        try:
            t_planner_odom = self.tf_buffer.lookup_transform(self.planner_frame, odom_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {odom_frame} to {self.planner_frame}: {ex}",
                throttle_duration_sec=self.throttle_duration,
            )
            return None

        state_untransformed = self.state_from_odom(msg)
        ego_vehicle_state = self.transform_state(state_untransformed, t_planner_odom)
        self.foresee_the_unseen_planner.update_state(ego_vehicle_state)

    # MARKER FUNCTIONS

    def get_ego_vehicle_marker(self):
        """Draw the ego vehicle bounding box"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.base_frame)
        ego_vehicle_bb = Marker(
            header=header,
            id=0,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(position=Point(**dict(zip(XYZ, self.ego_vehicle_offset)))),
            scale=Vector3(**dict(zip(XYZ, self.ego_vehicle_size))),
            color=ColorRGBA(**TRANSPARENT_BLUE),
            frame_locked=True,
            ns="ego_vehicle",
        )

        return [ego_vehicle_bb]

    def get_road_structure_marker(self):
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
                color=ColorRGBA(**BLACK),
                ns="road structure",
            )

            markers_list.append(lanelet_marker)
        return markers_list

    def get_filter_polygon_marker(self):
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
            color=ColorRGBA(**BLUE),
            ns="experiment env",
        )
        return [filter_marker]

    def get_fov_marker(self, sensor_view: Optional[ShapelyPolygon] = None):
        if sensor_view is not None:
            x, y = sensor_view.exterior.coords.xy
            points = np.asarray(tuple(zip(x, y)), dtype=np.float_)
            point_list = []
            for point in [*points, points[0]]:
                point_list.append(Point(**dict(zip(XYZ, np.array(point, dtype=np.float_)))))
        else:
            point_list = []

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        fov_marker = Marker(
            header=header,
            id=0,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            points=point_list,
            scale=Vector3(x=0.01),
            color=ColorRGBA(**BLUE),
            frame_locked=True,
            ns="fov",
        )
        return [fov_marker]

    def get_goal_marker(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        goal_marker = Marker(
            header=header,
            type=Marker.POINTS,
            action=Marker.MODIFY,
            id=0,
            points=[
                Point(x=float(self.configuration.get("goal_point_x")), y=float(self.configuration.get("goal_point_y")))
            ],
            colors=[ColorRGBA(g=1.0, a=1.0)],
            scale=Vector3(x=0.06, y=0.06),
            ns="goal",
        )
        return [goal_marker]

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

    def get_trajectory_marker(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        marker_list = []
        if self.foresee_the_unseen_planner.trajectory is not None:
            point_list = []
            color_rgba_list = []
            cmap = mpl.cm.get_cmap("cool")
            # o in the list below is of type InitialState
            positions = [o.position for o in self.foresee_the_unseen_planner.trajectory.trajectory.state_list]
            for x, y in positions:
                point_list.append(Point(x=x, y=y))
                r, g, b, a = cmap(len(point_list) / len(positions))
                color_rgba_list.append(ColorRGBA(r=r, g=g, b=b, a=a))
            trajectory_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=0,
                points=point_list,
                colors=color_rgba_list,
                scale=Vector3(x=0.03, y=0.03),
                ns="trajectory",
            )
            marker_list.append(trajectory_marker)
        if self.foresee_the_unseen_planner.planner.waypoints is not None:
            point_list = []
            color_rgba_list = []
            for x, y in self.foresee_the_unseen_planner.planner.waypoints:
                point_list.append(Point(x=float(x), y=float(y)))
                color_rgba_list.append(ColorRGBA(**BLACK))
            waypoints_marker = Marker(
                header=header,
                type=Marker.POINTS,
                action=Marker.MODIFY,
                id=1,
                points=point_list,
                colors=color_rgba_list,
                scale=Vector3(x=0.03, y=0.03),
                ns="waypoints",
            )
            marker_list.append(waypoints_marker)

        return marker_list

    def polygons_to_ros_marker(
        self,
        polygons: Union[np.ndarray, List[np.ndarray]],
        color: Color,
        namespace: str,
        marker_id: int = 0,
        use_triangulation: bool = False,
        z: float = 0.0,
    ) -> Marker:
        """Converts a array of polygon which is defined by a numpy array (N, 2) to ROS marker"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        if use_triangulation:  # make filled polygons with triangles
            point_type_list, colors = [], []
            for polygon in polygons:
                if len(polygon) < 4:
                    continue

                polygon = remove_redudant_vertices_polygon(polygon)
                try:
                    for t in triangulate(polygon[:-1]):  # do not include start point twice
                        for x, y in t:
                            point_type_list.append(Point(x=float(x), y=float(y), z=float(z)))
                        colors.append(ColorRGBA(**color))
                except ValueError as e:
                    self.get_logger().warn(f"triangulation failed: {e}")

            marker = Marker(
                header=header,
                id=marker_id,
                type=Marker.TRIANGLE_LIST,
                action=Marker.ADD,
                points=point_type_list,
                colors=colors,
                scale=Vector3(x=1.0, y=1.0, z=1.0),
                color=ColorRGBA(a=color.get("a", 1.0)),
                ns=namespace,
            )
            markers = []
        else:  # plot polygons as lines
            markers = []
            for idx, polygon in enumerate(polygons):
                point_type_list = []
                for x, y in polygon:
                    point_type_list.append(Point(x=float(x), y=float(y), z=float(z)))

                marker = Marker(
                    header=header,
                    id=marker_id + idx,
                    type=Marker.LINE_STRIP,
                    action=Marker.ADD,
                    points=point_type_list,
                    scale=Vector3(x=0.1),
                    color=ColorRGBA(**color),
                    ns=namespace,
                )
                markers.append(marker)
        return markers

    def get_shadow_marker(self, shadow_obstacles: Optional[List[Obstacle]] = None):
        """Makes a marker for the shadow obstacles and their prediction."""
        if shadow_obstacles is None:
            return []

        markers = []

        N = self.predictions_to_visualize
        steps = np.arange(0, N * self.configuration["prediction_step_size"], self.configuration["prediction_step_size"])
        steps = np.clip(steps, 0, self.configuration["prediction_horizon"] - 1)
        # occupancy per prediction step
        pred_polygons_per_step = [
            [s.prediction.occupancy_set[i].shape.vertices for s in shadow_obstacles] for i in steps
        ]

        base_color = np.array([SHADOW_PRED_COLOR[key] for key in ["r", "g", "b"]])
        diff_color = 1 - base_color
        for idx, pred_polygons in enumerate(pred_polygons_per_step):
            color = base_color + diff_color * idx / N
            color_dict = {key: float(value) for key, value in zip(["r", "g", "b"], color)}
            color_dict['a'] = 1.
            markers.extend(
                self.polygons_to_ros_marker(
                    polygons=pred_polygons,
                    color=color_dict,
                    namespace="shadow obstacles prediction",
                    marker_id=idx * 10000,  # in case there are multiple polygons
                    use_triangulation=self.use_triangulation,
                    z=-(idx + 1) * 0.01,
                )
            )

        polygons_occluded_set = [s.obstacle_shape.vertices for s in shadow_obstacles]
        markers.extend(
            self.polygons_to_ros_marker(
                polygons=polygons_occluded_set,
                color=SHADOW_OBS_COLOR,
                namespace="shadow obstacles",
                use_triangulation=self.use_triangulation,
                z=0.0,
            )
        )

        return markers

    # OTHER

    def filter_pointcloud(self, scan_msg: LaserScan):
        """Filter points from the pointcloud from the lidar to reduce the computation of the `datmo` package."""
        # Convert the ranges to a pointcloud
        laser_frame = scan_msg.header.frame_id
        points_laser = self.laserscan_to_pointcloud(scan_msg)
        points_laser = np.nan_to_num(points_laser, nan=1e99, posinf=1e99, neginf=-1e99)
        try:
            points_map = self.transform_pointcloud(points_laser, self.map_frame, laser_frame)
        except TransformException:
            return None

        # filter the points based on a polygon
        A, B = self.laser_filter_matrices
        # mask_polygon = np.all(A @ points_map.T <= np.repeat(B.reshape(-1, 1), points_map.shape[0], axis=1), axis=0)
        mask_polygon = np.all(A @ points_map.T <= B.reshape(-1, 1), axis=0)

        # filter the points based on a circle -> distance from origin
        mask_fov = np.array(scan_msg.ranges) <= self.configuration.get("view_range")

        # Combine masks and make new message
        mask = mask_polygon & mask_fov
        new_msg = scan_msg
        ranges = np.array(scan_msg.ranges).astype(np.float32)
        ranges[~mask] = np.inf  # set filtered out values to infinite
        new_msg.ranges = ranges

        self.filtered_laser_publisher.publish(new_msg)

    def FOV_from_laser(self, scan_msg: LaserScan):
        """Determines the FOV based on the LaserScan message"""
        laser_frame = scan_msg.header.frame_id
        points_laser = self.laserscan_to_pointcloud(scan_msg, max_range=self.configuration.get("view_range"))

        try:
            points_planner = self.transform_pointcloud(points_laser, self.planner_frame, laser_frame)
            # subsample the points to make the shape simpler
            points_planner = points_planner[::10]
            sensor_FOV = ShapelyPolygon(points_planner) if len(points_planner) >= 3 else None
        except TransformException:
            sensor_FOV = None
        self.foresee_the_unseen_planner.update_fov(sensor_FOV)

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

    @staticmethod
    def transform_state(state: State, transform: TransformStamped) -> State:
        """Converts a Commonroad state (2D position, orientation and scalar velocity) based on a ROS transform"""
        t_matrix = matrix_from_transform(transform)
        position = np.hstack((state.position, [0, 1]))  # 4D position
        state.position = (t_matrix @ position)[:2]
        r_matrix = t_matrix[:2, :2]
        state.orientation = np.arctan2(
            *np.flip(r_matrix @ np.array([np.cos(state.orientation), np.sin(state.orientation)]))
        )

        return state

    @staticmethod
    def laserscan_to_pointcloud(scan_msg: LaserScan, max_range: Optional[float] = None) -> np.ndarray:
        """Converts a ROS LaserScan message to a numpy array with 2D points with shape: (N, 2)"""
        ranges = np.array(scan_msg.ranges)
        if max_range is not None:
            ranges[ranges > max_range] = max_range
        N = len(ranges)
        angles = scan_msg.angle_min + np.arange(N) * scan_msg.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        return (ranges * cos_sin_map).T  # 2D points

    def transform_pointcloud(self, points: np.ndarray, target_frame: str, source_frame: str) -> np.ndarray:
        """Converts a pointcloud of 2D points (np.ndarray with shape = (N, 2)) with a ROS transform.
        Raises a TransformException when the transform is not available"""
        assert points.shape[1] == 2, "Should be 2D points with array shape (N, 2)"
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {source_frame} to {target_frame}: {ex}",
                throttle_duration_sec=self.throttle_duration,
            )
            raise TransformException
        points_4D = np.hstack((points, np.zeros((len(points), 1)), np.ones((len(points), 1))))
        t_mat = matrix_from_transform(t)
        return (t_mat @ points_4D.T)[:2].T


def main(args=None):
    rclpy.init(args=args)

    planner_node = PlannerNode()

    rclpy.spin(planner_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
