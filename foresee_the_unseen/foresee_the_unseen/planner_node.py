import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from setuptools import find_packages
from ament_index_python import get_package_share_directory
from typing import Optional, List, Union, Tuple, Dict, TypedDict

from commonroad.scenario.state import InitialState, State
from commonroad.scenario.obstacle import Obstacle, ObstacleType, DynamicObstacle
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import SetBasedPrediction
from commonroad.scenario.trajectory import Trajectory as TrajectoryCR

from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

from builtin_interfaces.msg import Time as TimeMsg

# from datmo.msg import TrackArray
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import (
    Vector3,
    Point,
    Point32,
    PolygonStamped,
    TransformStamped,
    Pose,
    PoseStamped,
    Quaternion,
    Twist,
    Accel,
)
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import ParameterDescriptor
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg, ProjectedOccludedArea

from tf2_ros import TransformException  # type: ignore
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PolygonStamped  # necessary to enable Buffer.transform

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    euler_from_quaternion,
)
from foresee_the_unseen.lib.foresee_the_unseen import ForeseeTheUnseen, NoUpdatePossible
from foresee_the_unseen.lib.triangulate import triangulate, remove_redudant_vertices_polygon


Color = TypedDict("Color", {"r": float, "g": float, "b": float, "a": float})
RED: Color = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
GREEN: Color = {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0}
BLUE: Color = {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
YELLOW: Color = {"r": 1.0, "g": 1.0, "b": 0.0, "a": 1.0}
BLACK: Color = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
TRANSPARENT_BLACK: Color = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.3}
TRANSPARENT_ORANGE: Color = {"r": 0.93, "g": 0.54, "b": 0.15, "a": 0.7}
TRANSPARENT_BLUE: Color = {"r": 0.0, "g": 0.0, "b": 1.0, "a": 0.6}
TRANSPARENT_GREY: Color = {"r": 0.5, "g": 0.5, "b": 0.5, "a": 0.8}

SHADOW_OBS_COLOR: Color = {"r": 100.0 / 255.0, "g": 0.0, "b": 0.0, "a": 1.0}
SHADOW_PRED_COLOR: Color = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}

# TODO: remove approaches that rely on this to speed up the code
XYZ = ["x", "y", "z"]


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner_node")
        self.throttle_duration = 3  # s

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "laser")
        self.declare_parameter("planner_frame", "planner")

        self.declare_parameter("fov_topic", "fov")
        self.declare_parameter("odom_topic", "odom")
        # self.declare_parameter("datmo_topic", "datmo/box_kf")
        self.declare_parameter("visualization_topic", "visualization/planner")
        self.declare_parameter("trajectory_topic", "trajectory")
        self.declare_parameter("startup_topic", "/goal_pose")
        self.declare_parameter("projected_area_topic", "/projected_occluded_area")

        self.declare_parameter(
            "road_xml",
            os.path.join(
                get_package_share_directory("foresee_the_unseen"), "resource/road_structure_15_reduced_points.xml"
            ),
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
        self.declare_parameter("error_models_directory", "none")

        self.declare_parameter("ego_vehicle_size", [0.3, 0.18, 0.12])  # L x W x H [m]
        self.declare_parameter("ego_vehicle_offset", [0.0, 0.0, 0.06])  # L x W x H [m]
        self.declare_parameter("planner_frequency", 2.0)

        self.declare_parameter("use_triangulation", True)  # for the visualization of the polygons
        self.declare_parameter("num_pred_to_visualize", 5)
        self.declare_parameter("do_visualize", True)

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.planner_frame = self.get_parameter("planner_frame").get_parameter_value().string_value

        self.fov_topic = self.get_parameter("fov_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        # self.datmo_topic = self.get_parameter("datmo_topic").get_parameter_value().string_value
        self.visualization_topic = self.get_parameter("visualization_topic").get_parameter_value().string_value
        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.startup_topic = self.get_parameter("startup_topic").get_parameter_value().string_value
        self.project_area_topic = self.get_parameter("projected_area_topic").get_parameter_value().string_value

        self.road_xml = self.get_parameter("road_xml").get_parameter_value().string_value
        self.config_yaml = self.get_parameter("foresee_the_unseen_yaml").get_parameter_value().string_value
        self.log_root_directory = self.get_parameter("log_directory").get_parameter_value().string_value
        self.log_root_directory = None if self.log_root_directory == "none" else self.log_root_directory
        self.error_models_directory = self.get_parameter("error_models_directory").get_parameter_value().string_value
        self.error_models_directory = None if self.error_models_directory == "none" else self.error_models_directory
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
        self.foresee_the_unseen_timer = self.create_timer(1 / self.frequency, self.plan)
        self.foresee_the_unseen_timer.cancel()

        # Transformer listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers and publishers
        self.create_subscription(PolygonStamped, self.fov_topic, self.fov_callback, 1)  # Laser scan
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)  # Ego vehicle pose
        # self.create_subscription(TrackArray, self.datmo_topic, self.datmo_callback, 5)
        self.create_subscription(PoseStamped, self.startup_topic, self.startup_callback, 1)
        self.marker_array_publisher = self.create_publisher(MarkerArray, self.visualization_topic, 2)
        self.trajectory_publisher = self.create_publisher(TrajectoryMsg, self.trajectory_topic, 1)
        self.project_area_publisher = self.create_publisher(ProjectedOccludedArea, self.project_area_topic, 5)

        self.foresee_the_unseen_planner = ForeseeTheUnseen(
            config_yaml=self.config_yaml,
            road_xml=self.road_xml,
            frequency=self.frequency,
            logger=self.get_logger(),  # type: ignore -- apply ducktyping principle for the logger
            log_dir=self.log_root_directory,
            error_models_dir=self.error_models_directory,
        )

    # CALLBACKS
    def plan(self):
        """Calls the Foresee The Unseen Planner and updates the visualization"""
        start_time = time.time()
        planning_time = self.get_clock().now()
        try:
            shadow_obstacles, sensor_view, trajectory, no_stop_zone, prediction, projected_area = (
                self.foresee_the_unseen_planner.update_scenario(planning_time.nanoseconds * 1e-9)
            )
        except NoUpdatePossible:
            self.get_logger().warn("No planner update step possible", throttle_duration_sec=self.throttle_duration)
            shadow_obstacles, sensor_view, trajectory, no_stop_zone, prediction, projected_area = (None,)*6

        if trajectory is not None:
            self.publish_trajectory(trajectory)

        if projected_area is not None:
            area_msg = ProjectedOccludedArea()
            area_msg.header.stamp = self.get_clock().now().to_msg()
            area_msg.area.data = float(projected_area)
            self.project_area_publisher.publish(area_msg)

        start_viz_time = time.time()
        if self.do_visualize:
            self.visualization_callback(shadow_obstacles, sensor_view, trajectory, no_stop_zone, prediction)

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

    @staticmethod
    def quaternion_from_yaw(yaw):
        return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    def publish_trajectory(self, trajectory: TrajectoryCR) -> None:
        """Publishes the Commonroad trajectory on a topic for the trajectory follower node."""
        # start_stamp = self.get_clock().now()
        header_path = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        pose_stamped_list = []
        # init_time_step = trajectory.state_list[0].time_step - 1
        for state in trajectory.state_list:
            quaternion = Quaternion(
                **{k: v for k, v in zip(["x", "y", "z", "w"], self.quaternion_from_yaw(state.orientation))}
            )
            position = Point(x=float(state.position[0]), y=float(state.position[1]))
            pose_stamp = Time(seconds=state.time_step).to_msg()
            # time_diff = (state.time_step - init_time_step) * 1 / self.frequency
            # header_pose = Header(
            #     stamp=(start_stamp + Duration(seconds=int(time_diff), nanoseconds=int((time_diff % 1) * 1e9))).to_msg()
            # )
            header_pose = Header(stamp=pose_stamp)
            pose_stamped_list.append(
                PoseStamped(header=header_pose, pose=Pose(position=position, orientation=quaternion))
            )
        twist_list = [Twist(linear=Vector3(x=float(s.velocity))) for s in trajectory.state_list]
        if trajectory.state_list[0].acceleration is not None:
            accel_list = [Accel(linear=Vector3(x=float(s.acceleration))) for s in trajectory.state_list]
        else:
            accel_list = []
        trajectory_msg = TrajectoryMsg(
            path=Path(header=header_path, poses=pose_stamped_list), velocities=twist_list, accelerations=accel_list
        )

        self.trajectory_publisher.publish(trajectory_msg)

    def startup_callback(self, msg: PoseStamped) -> None:
        """This function allows the robot to start moving."""
        self.foresee_the_unseen_timer.reset()

    def fov_callback(self, fov_msg: PolygonStamped) -> None:
        try:
            if fov_msg.header.frame_id != self.planner_frame:
                fov_msg = self.tf_buffer.transform(fov_msg, self.planner_frame)  # type: ignore
            fov_points = np.array([[p.x, p.y] for p in fov_msg.polygon.points])
            if len(fov_points) >= 3:
                polygon = ShapelyPolygon(fov_points)
                if polygon.is_valid:
                    self.foresee_the_unseen_planner.set_field_of_view(
                        polygon, fov_msg.header.stamp.sec + fov_msg.header.stamp.nanosec * 1e-9
                    )
        except TransformException as ex:
            self.get_logger().info(str(ex), throttle_duration_sec=self.throttle_duration)

    def visualization_callback(
        self,
        shadow_obstacles: Optional[List[DynamicObstacle]] = None,
        sensor_view: Optional[ShapelyPolygon] = None,
        trajectory: Optional[TrajectoryCR] = None,
        no_stop_zone: Optional[DynamicObstacle] = None,
        prediction: Optional[SetBasedPrediction] = None,
    ):
        # TODO: Only remove markers that need to and only add markers that need to be added every time step
        markers = []
        markers.append(Marker(action=Marker.DELETEALL))  # remove all markers
        markers += self.get_ego_vehicle_marker()
        markers += self.get_road_structure_marker()
        markers += self.get_goal_marker()

        if sensor_view is not None:
            markers += self.get_fov_marker(sensor_view)
        if trajectory is not None:
            markers += self.get_trajectory_marker(trajectory)
        if shadow_obstacles is not None:
            markers += self.get_shadow_marker(shadow_obstacles)
        if no_stop_zone is not None:
            markers += self.get_no_stop_zone_marker(no_stop_zone)
        if prediction is not None:
            markers += self.get_prediction_marker(prediction)
        else:
            prediction = self.foresee_the_unseen_planner.planner.fastest_prediction
            if prediction is not None:
                markers += self.get_prediction_marker(prediction, color=TRANSPARENT_ORANGE)

        self.marker_array_publisher.publish(MarkerArray(markers=markers))

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
        self.foresee_the_unseen_planner.set_ego_vehicle_state(
            ego_vehicle_state, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )

    # MARKER FUNCTIONS

    def get_ego_vehicle_marker(self) -> List[Marker]:
        """Get the marker of the box representing the ego vehicle"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.base_frame)
        ego_vehicle_bb = Marker(
            header=header,
            id=0,
            type=Marker.CUBE,
            action=Marker.ADD,
            # pose=Pose(position=Point(**dict(zip(XYZ, self.ego_vehicle_offset)))),
            pose=Pose(
                position=Point(x=self.ego_vehicle_offset[0], y=self.ego_vehicle_offset[1], z=self.ego_vehicle_offset[2])
            ),
            # scale=Vector3(**dict(zip(XYZ, self.ego_vehicle_size))),
            scale=Vector3(x=self.ego_vehicle_size[0], y=self.ego_vehicle_size[1], z=self.ego_vehicle_size[2]),
            color=ColorRGBA(**TRANSPARENT_BLUE),
            frame_locked=True,
            ns="ego_vehicle",
        )

        return [ego_vehicle_bb]

    def get_road_structure_marker(self) -> List[Marker]:
        """Get the markers that visualize the road boundaries."""
        markers = []
        for lanelet_id, points in self.road_polygons.items():
            point_type_list = []
            points = points
            for point in points:
                # point_type_list.append(Point(**dict(zip(XYZ, point))))
                point_type_list.append(Point(x=point[0], y=point[1]))

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

            markers.append(lanelet_marker)
        return markers

    def get_fov_marker(self, sensor_view: Optional[ShapelyPolygon] = None) -> List[Marker]:
        """Get the marker that visualizes the field of view (FOV)"""
        if isinstance(sensor_view, ShapelyPolygon):
            x, y = sensor_view.exterior.coords.xy
            points = np.asarray(tuple(zip(x, y)), dtype=np.float_)
            point_list = (
                [Point(x=p[0], y=p[1]) for p in np.append(points, points[0:1], axis=0)]
                if points.size and points.ndim == 2
                else []
            )
        elif isinstance(sensor_view, ShapelyMultiPolygon):
            point_list = []
            for polygon in sensor_view:
                x, y = polygon.exterior.coords.xy
                points = np.asarray(tuple(zip(x, y)), dtype=np.float_)
                point_list.extend(
                    [Point(x=p[0], y=p[1]) for p in np.append(points, points[0:1], axis=0)]
                    if points.size and points.ndim == 2
                    else []
                )
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

    def get_goal_marker(self) -> List[Marker]:
        """Get the marker for the goal position"""
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

    def get_trajectory_marker(self, trajectory: TrajectoryCR) -> List[Marker]:
        """Get the markers for the Commonroad trajectory and the waypoints found by the planner"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        marker_list = []
        if trajectory is not None:
            point_list = []
            color_rgba_list = []
            cmap = plt.get_cmap("cool")
            # o in the list below is of type InitialState
            positions = [o.position for o in trajectory.state_list]
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
        linewidth: float = 0.1,
        z: float = 0.0,
    ) -> List[Marker]:
        """Makes ROS marker messages based on an array of polygons where each polygon is defined by a numpy array (N, 2)"""
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.planner_frame)
        markers = []
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
                    self.get_logger().warn(f"triangulation failed: {e}", throttle_duration_sec=self.throttle_duration)

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
            markers.append(marker)
        else:  # plot polygons as lines
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
                    scale=Vector3(x=float(linewidth)),
                    color=ColorRGBA(**color),
                    ns=namespace,
                )
                markers.append(marker)
        return markers

    def get_shadow_marker(self, shadow_obstacles: List[DynamicObstacle]) -> List[Marker]:
        """Makes a marker for the shadow obstacles and their prediction."""
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
            color_dict["a"] = 1.0
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

    def get_no_stop_zone_marker(self, zone: Obstacle) -> List[Marker]:
        """Visualize the no stop zone at an intersection"""
        polygon = zone.obstacle_shape.vertices
        return self.polygons_to_ros_marker(
            polygons=[polygon], color=YELLOW, namespace="no stop zone", linewidth=0.05, z=0.1
        )

    def get_prediction_marker(self, prediction: SetBasedPrediction, color: Color = TRANSPARENT_BLACK) -> List[Marker]:
        """Visualize the prediction of the future occupancies"""
        polygons = [o.shape.vertices for o in prediction.occupancy_set]
        markers_lines = self.polygons_to_ros_marker(
            polygons=polygons,
            color=color,
            namespace="set based prediction",
            use_triangulation=False,
            linewidth=0.01,
            z=0.01,
        )
        if self.use_triangulation:
            markers_filled = self.polygons_to_ros_marker(
                polygons=polygons,
                color=color,
                namespace="set based prediction",
                use_triangulation=True,
                z=0.01,
            )
            markers_lines.extend(markers_filled)

        return markers_lines

    # OTHER
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
        velocity = odom.twist.twist.linear.x
        return InitialState(
            position=np.array(position),
            orientation=yaw,  # along x is 0, ccw is positive
            velocity=velocity,
            time_step=time_step,
        )

    @staticmethod
    def transform_state(state: InitialState, transform: TransformStamped) -> InitialState:
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
