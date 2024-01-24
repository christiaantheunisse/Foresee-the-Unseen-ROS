import rclpy
import numpy as np
from rclpy.node import Node
from setuptools import find_packages

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3, Point, Polygon, PolygonStamped
from collections.abc import Sequence

from foresee_the_unseen.lib.helper_functions import polygons_from_road_xml

RGBA = ["r", "g", "b", "a"]
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.0, 1.0, 1.0]
TRANSPARENT_BLUE = [0.0, 0.0, 1.0, 0.6]
BLACK = [0.0, 0.0, 0.0, 1.0]
TRANSPARENT_GREY = [0.5, 0.5, 0.5, 0.8]

XYZ = ["x", "y", "z"]


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        self.publisher_marker_array = self.create_publisher(MarkerArray, "foresee_the_unseen_markers", 10)

        # Add ego vehicle bounding box
        self.ego_vehicle_size = [0.3, 0.18, 0.12]  # L x W x H [m]
        self.ego_vehicle_offset = [0.0, 0.0, self.ego_vehicle_size[2] / 2]  # [m]

        # TODO: Make these into parameters
        self.world_frame = "map"
        self.vehicle_frame = "base_link"
        self.road_xml = "/home/ubuntu/thesis_ws/src/foresee_the_unseen/resource/road_structure.xml"
        self.road_offset = np.array([0, 2], dtype=np.float_)

        # get road_structure
        self.road_polygons = polygons_from_road_xml(self.road_xml, self.road_offset)

        self.timer = self.create_timer(1, self.timer_callback)

        # TODO: Assign unique IDs
        self.ego_vehicle_id = 0
        # lanelets use the lanelet ID
        self.pot_ids = [10000, 10001]

    def timer_callback(self):
        markers = []
        markers += self.get_ego_vehicle_markers()
        markers += self.get_road_structure_markers()
        markers += self.get_flower_pots_markers()

        self.publisher_marker_array.publish(MarkerArray(markers=markers))


    def get_ego_vehicle_markers(self):
        """Draw the ego vehicle bounding box"""

        ego_vehicle_bb = Marker()

        ego_vehicle_bb.header.frame_id = self.vehicle_frame
        ego_vehicle_bb.id = self.ego_vehicle_id
        ego_vehicle_bb.type = Marker.CUBE
        ego_vehicle_bb.action = Marker.ADD

        ego_vehicle_bb.pose.position = Point(**dict(zip(XYZ, self.ego_vehicle_offset)))
        ego_vehicle_bb.scale = Vector3(**dict(zip(XYZ, self.ego_vehicle_size)))
        ego_vehicle_bb.color = ColorRGBA(**dict(zip(RGBA, TRANSPARENT_BLUE)))
        ego_vehicle_bb.frame_locked = True

        return [ego_vehicle_bb]

    def get_road_structure_markers(self):
        markers_list = []
        for lanelet_id, points in self.road_polygons.items():
            point_type_list = []
            points = points / 20
            for point in points:
                point_type_list.append(Point(**dict(zip(XYZ, point))))

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame)
            lanelet_marker = Marker(
                header=header,
                id=int(lanelet_id),
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                points=point_type_list,
                scale=Vector3(x=0.005),
                color=ColorRGBA(**dict(zip(RGBA, BLACK))),
            )

            markers_list.append(lanelet_marker)

        return markers_list

    def get_flower_pots_markers(self):
        left_pot, right_pot = Marker(), Marker()
        size = np.array([0.4, 0.4, 1], dtype=np.float_)
        for id_pot, pot in zip(self.pot_ids, [left_pot, right_pot]):
            pot.header.frame_id = self.world_frame
            pot.id = id_pot
            pot.type = Marker.CUBE
            pot.action = Marker.ADD
            pot.scale = Vector3(**dict(zip(XYZ, size)))
            pot.color = ColorRGBA(**dict(zip(RGBA, TRANSPARENT_GREY)))
            
        left_position = (np.array([1, 0.525, 0]) + [0, size[1]/2, size[2]/2]).astype(np.float_)
        right_position = (np.array([1, -0.525, 0]) + [0, -size[1]/2, size[2]/2]).astype(np.float_)
        left_pot.pose.position = Point(**dict(zip(XYZ, left_position)))
        right_pot.pose.position = Point(**dict(zip(XYZ, right_position)))

        return [left_pot, right_pot]


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
