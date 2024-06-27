import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.time import Duration

from sensor_msgs.msg import LaserScan

# ros2 run tf2_ros static_transform_publisher --frame-id base_link --child-frame-id laser


class LaserScanSensorNode(Node):
    """Act as a sensor that periodically sends LaserScan messages"""

    def __init__(self):
        super().__init__("laserscan_sensor_node")

        self.declare_parameter("scan_topic", "scan")
        self.declare_parameter("laser_frame", "laser")
        self.declare_parameter("frequency", 7.8)
        self.declare_parameter("number_of_points", 1080)
        self.declare_parameter("angle_min", -np.pi)
        self.declare_parameter("angle_max", np.pi)
        self.declare_parameter("range_max", 12.0)
        self.declare_parameter("range_min", 0.15)

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.laser_frame = self.get_parameter("laser_frame").get_parameter_value().string_value
        self.frequency = self.get_parameter("frequency").get_parameter_value().double_value
        self.number_of_points = self.get_parameter("number_of_points").get_parameter_value().integer_value
        self.angle_min = self.get_parameter("angle_min").get_parameter_value().double_value
        self.angle_max = self.get_parameter("angle_max").get_parameter_value().double_value
        self.range_max = self.get_parameter("range_max").get_parameter_value().double_value
        self.range_min = self.get_parameter("range_min").get_parameter_value().double_value

        self.fixed_sensor_value = self.get_laserscan_message()

        self.create_timer(1 / self.frequency, self.publish_laserscan_callback)

        self.laserscan_publisher = self.create_publisher(LaserScan, self.scan_topic, 5)

    def get_laserscan_message(self) -> LaserScan:
        """Get a LaserScan message that detected the maximum range at every angle"""
        msg = LaserScan()
        msg.header.stamp = (self.get_clock().now() - Duration(seconds=1/self.frequency + 0.1)).to_msg()
        msg.header.frame_id = self.laser_frame

        msg.angle_min = self.angle_min
        msg.angle_increment = (self.angle_max - self.angle_min) / (self.number_of_points)
        msg.angle_max = self.angle_max - msg.angle_increment
        msg.time_increment = 1 / self.frequency / self.number_of_points
        msg.scan_time = 1 / self.frequency
        msg.range_min = self.range_min
        msg.range_max = self.range_max
        msg.ranges = np.full(self.number_of_points, self.range_max)
        msg.intensities = np.full(self.number_of_points, 100.0)

        return msg

    def publish_laserscan_callback(self) -> None:
        """Periodically publish a fixed sensor value"""
        self.fixed_sensor_value.header.stamp = (self.get_clock().now() - Duration(seconds=1/self.frequency + 0.1)).to_msg()
        self.laserscan_publisher.publish(self.fixed_sensor_value)


def main(args=None):
    rclpy.init(args=args)

    laserscan_sensor_node = LaserScanSensorNode()

    rclpy.spin(laserscan_sensor_node)
    laserscan_sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
