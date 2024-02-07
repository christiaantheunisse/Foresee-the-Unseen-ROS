import rclpy
import numpy as np
from rclpy.node import Node
from setuptools import find_packages
from ament_index_python import get_package_share_directory
import os

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3, Point, Point32, PolygonStamped
from sensor_msgs.msg import LaserScan
from rcl_interfaces.msg import ParameterDescriptor

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    matrices_from_cw_cvx_polygon,
)

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
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "laser")

        self.declare_parameter("lidar_topic", "scan")
        self.declare_parameter("filtered_lidar_topic", "scan/road_env")
        self.declare_parameter("fov_topic", "visualization/fov")
        self.declare_parameter("obstacles_topic", "visualization/obstacles")

        self.declare_parameter(
            "road_xml",
            os.path.join(get_package_share_directory("foresee_the_unseen"), "resource/road_structure.xml"),
            ParameterDescriptor(
                description="The file name of the .xml file in the resources folder describing the road structure"
            ),
        )
        self.declare_parameter("road_translation", [0., 0.], ParameterDescriptor(description="translation of the road"))
        self.declare_parameter(
            "road_rotation", 0., ParameterDescriptor(description="CCW rotation of the road structure")
        )
        self.declare_parameter(
            "environment_boundary",
            [2.0, 3.0, 2.0, 0.0, -2.0, 0.0, -2.0, 3.0],
            ParameterDescriptor(
                description="Convex polygon describing part of the map frame visuable to the lidar of the ego_vehicle"
                "in the foresee-the-unseen algorithm. [x1, y1, x2, y2, ...]"
            ),
        )
        self.declare_parameter("ego_vehicle_size", [0.3, 0.18, 0.12]) # L x W x H [m]
        self.declare_parameter("ego_vehicle_offset", [0., 0., 0.06]) # L x W x H [m]


        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.filtered_lidar_topic = self.get_parameter("filtered_lidar_topic").get_parameter_value().string_value
        self.fov_topic = self.get_parameter("fov_topic").get_parameter_value().string_value
        self.markersarray_topic = self.get_parameter("obstacles_topic").get_parameter_value().string_value
        self.road_xml = self.get_parameter("road_xml").get_parameter_value().string_value
        self.road_translation = np.array(
            self.get_parameter("road_translation").get_parameter_value().double_array_value
        )
        self.road_rotation = self.get_parameter("road_rotation").get_parameter_value().double_value
        self.filter_polygon = np.array(
            self.get_parameter("environment_boundary").get_parameter_value().double_array_value
        ).reshape(-1, 2)
        self.ego_vehicle_size = self.get_parameter("ego_vehicle_size").get_parameter_value().double_array_value
        self.ego_vehicle_offset = self.get_parameter("ego_vehicle_offset").get_parameter_value().double_array_value

        # MarkerArray publisher -> maybe to different speeds
        self.publisher_marker_array = self.create_publisher(MarkerArray, self.markersarray_topic, 10)

        # get road_structure
        self.road_polygons = polygons_from_road_xml(self.road_xml, self.road_translation, self.road_rotation)

        # Timer for the visualization -> different timers for fov and other things
        self.timer = self.create_timer(1, self.timer_callback)

        # constrian the pointcloud to some box
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.laser_subscription = self.create_subscription(LaserScan, self.lidar_topic, self.laser_callback, 5)
        self.laser_publisher = self.create_publisher(LaserScan, self.filtered_lidar_topic, 5)
        self.laser_filter_matrices = matrices_from_cw_cvx_polygon(self.filter_polygon)

        # FOV publisher
        # self.fov_publisher = self.create_publisher(PolygonStamped, self.fov_topic, 1)

    def timer_callback(self):
        markers = []
        markers += self.get_ego_vehicle_markers()
        markers += self.get_road_structure_markers()
        markers += self.get_flower_pots_markers()
        markers += self.get_filter_polygon()
        markers += self.get_fov_marker()

        self.publisher_marker_array.publish(MarkerArray(markers=markers))


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
            points = points / 20
            for point in points:
                point_type_list.append(Point(**dict(zip(XYZ, point))))

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)
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
        new_point_cloud = self.filter_pointcloud(msg)

    def filter_pointcloud(self, scan_msg: LaserScan):
        """Filter points from the pointcloud from the lidar to reduce the computation of the `datmo` package."""
        # TODO: transform the bounds instead of the laserscan itself

        # Convert the ranges to a pointcloud
        ranges = np.array(scan_msg.ranges)
        N = len(ranges)
        angles = scan_msg.angle_min + np.arange(N) * scan_msg.angle_increment
        cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
        points_laser = ranges * cos_sin_map  # 2D points
        points_laser = np.vstack((points_laser, np.zeros((1, len(angles)))))
        # points_laser = points_laser[((ranges > scan_msg.range_min) & (ranges < scan_msg.range_max))] # apply the bounds

        # Transform the pointcloud
        points_laser_4D = np.vstack((points_laser, np.full((1, points_laser.shape[1]), 1)))

        # Convert to map frame
        try:
            t_map_laser = self.tf_buffer.lookup_transform(self.map_frame, self.lidar_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f"Could not transform {self.lidar_frame} to {self.map_frame}: {ex}")
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
            self.get_logger().info(f"Could not transform {self.base_frame} to {self.map_frame}: {ex}")
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

        self.laser_publisher.publish(new_msg)

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
            ns="experiment env"
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
