import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
from typing import Union, Optional
from scipy.spatial.transform import Rotation, Slerp

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Transform, PoseStamped
from nav_msgs.msg import Odometry
from racing_bot_interfaces.msg import Trajectory

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose


class OdometrySensorNode(Node):
    """Act as a sensor that periodically sends Odometry messages and publish the necessary transforms"""

    def __init__(self):
        super().__init__("odometry_sensor_node")

        self.declare_parameter("odometry_topic", "odometry/filtered")
        self.declare_parameter("trajectory_topic", "trajectory")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("frequency", 30.0)

        self.odometry_topic = self.get_parameter("odometry_topic").get_parameter_value().string_value
        self.trajectory_topic = self.get_parameter("trajectory_topic").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.frequency = self.get_parameter("frequency").get_parameter_value().double_value

        self.trajectory: Optional[Trajectory] = None

        self.create_timer(1 / self.frequency, self.publish_odometry_callback)

        self.create_subscription(Trajectory, self.trajectory_topic, self.trajectory_callback, 5)
        self.odometry_publisher = self.create_publisher(Odometry, self.odometry_topic, 5)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish_odometry_callback(self) -> None:
        """Periodically publish the odometry. This is either the start position or based on a received trajectory.

        The trajectory is in the map_frame, so the odometry is also in the map frame. The transform based on the
        odometry should be from the base_frame to the odom_frame. The odom_frame and map_frame are similar, so a
        transform with all zeros can be published.
        """
        stamp = self.get_clock().now().to_msg()

        # publish odometry
        if self.trajectory is None:
            odometry = self.get_odometry_message(
                position=[0, 0], linear_velocity=0, orientation=0, angular_velocity=0, stamp=stamp
            )
        else:
            odometry = self.get_odometry_from_trajectory(trajectory=self.trajectory, stamp=stamp)
        self.odometry_publisher.publish(odometry)

        # broadcast transform from: odom_frame to base_frame
        t_odom_base = self.get_transform_message(odometry, src_frame=self.odom_frame, target_frame=self.base_frame)
        self.tf_broadcaster.sendTransform(t_odom_base)
        # broadcast (static) transform from: map_frame to odom_frame
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = self.get_clock().now().to_msg()
        t_map_odom.header.frame_id, t_map_odom.child_frame_id = self.map_frame, self.odom_frame
        self.tf_broadcaster.sendTransform(t_map_odom)

    def trajectory_callback(self, msg: Trajectory) -> None:
        """The trajectory topic callback. Convert the trajectory to the map frame."""
        if msg.path.header.frame_id != self.map_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame, msg.path.header.frame_id, Time())
                msg = self.transform_trajectory(msg, transform)
            except TransformException as ex:
                self.get_logger().info(str(ex))

        self.trajectory = msg

    @staticmethod
    def quaternion_from_yaw(yaw) -> list[float]:
        return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    @staticmethod
    def stamp_to_float(stamp: TimeMsg) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def transform_trajectory(trajectory: Trajectory, transform: TransformStamped) -> Trajectory:
        trajectory.path.poses = [
            PoseStamped(header=p.header, pose=do_transform_pose(p.pose, transform)) for p in trajectory.path.poses
        ]
        trajectory.path.header.frame_id = transform.header.frame_id
        return trajectory

    def get_odometry_from_trajectory(self, trajectory: Trajectory, stamp: TimeMsg) -> Odometry:
        stamp_f = OdometrySensorNode.stamp_to_float(stamp)
        traject_stamps_f = [OdometrySensorNode.stamp_to_float(p.header.stamp) for p in trajectory.path.poses]

        # find the indices of the poses to interpolate between
        insert_idx = np.searchsorted(traject_stamps_f, stamp_f)
        next_idx, prev_idx = min(insert_idx, len(traject_stamps_f) - 1), max(0, insert_idx - 1)
        next_st, prev_st = traject_stamps_f[next_idx], traject_stamps_f[prev_idx]
        interp_factor = (stamp_f - prev_st) / (next_st - prev_st) if not next_st == prev_st else 0.0
        odometry = Odometry()
        pose = odometry.pose.pose
        odometry.header.frame_id = trajectory.path.header.frame_id  # positions and orientations are in this frame
        odometry.header.stamp = stamp
        odometry.child_frame_id = self.base_frame  # velocities are in the base frame
        # interpolate the position
        next_position = trajectory.path.poses[next_idx].pose.position  # type: ignore
        prev_position = trajectory.path.poses[prev_idx].pose.position  # type: ignore
        pose.position.x = (1 - interp_factor) * prev_position.x + interp_factor * next_position.x
        pose.position.y = (1 - interp_factor) * prev_position.y + interp_factor * next_position.y

        # interpolate the orientation
        next_orient = [getattr(trajectory.path.poses[next_idx].pose.orientation, attr) for attr in ["x", "y", "z", "w"]]  # type: ignore
        prev_orient = [getattr(trajectory.path.poses[prev_idx].pose.orientation, attr) for attr in ["x", "y", "z", "w"]]  # type: ignore
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([prev_orient, next_orient]))
        obs_orient = slerp(interp_factor).as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = obs_orient

        # interpolate velocity
        if trajectory.velocities:
            next_velocity = trajectory.velocities[next_idx].linear.x  # type: ignore
            prev_velocity = trajectory.velocities[prev_idx].linear.x  # type: ignore
            odometry.twist.twist.linear.x = (1 - interp_factor) * prev_velocity + interp_factor * next_velocity

        return odometry

    def get_odometry_message(
        self,
        position: Union[np.ndarray, list],
        linear_velocity: float,
        orientation: float,
        angular_velocity: float,
        stamp: TimeMsg,
    ) -> Odometry:
        """Get the odometry message for the given position, velocity and orientation."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.base_frame

        msg.pose.pose.position.x = float(position[0])
        msg.pose.pose.position.y = float(position[1])

        quaternion = self.quaternion_from_yaw(orientation)
        msg.pose.pose.orientation.x = float(quaternion[0])
        msg.pose.pose.orientation.y = float(quaternion[1])
        msg.pose.pose.orientation.z = float(quaternion[2])
        msg.pose.pose.orientation.w = float(quaternion[3])

        msg.twist.twist.linear.x = float(linear_velocity)
        msg.twist.twist.angular.z = float(angular_velocity)

        return msg

    def get_transform_message(
        self,
        odometry: Odometry,
        src_frame: str,
        target_frame: str,
    ) -> TransformStamped:
        """Get a transform for the given position and orientation"""
        t = TransformStamped()
        t.header.stamp = odometry.header.stamp
        t.header.frame_id = src_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = odometry.pose.pose.position.x
        t.transform.translation.y = odometry.pose.pose.position.y

        t.transform.rotation.x = odometry.pose.pose.orientation.x
        t.transform.rotation.y = odometry.pose.pose.orientation.y
        t.transform.rotation.z = odometry.pose.pose.orientation.z
        t.transform.rotation.w = odometry.pose.pose.orientation.w

        return t


def main(args=None):
    rclpy.init(args=args)

    odometry_sensor_node = OdometrySensorNode()

    rclpy.spin(odometry_sensor_node)
    odometry_sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
