import rclpy
import rclpy.node as Node
import os
from ament_index_python import get_package_share_directory
import yaml
import copy
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
from commonroad.geometry.shape import Rectangle, Circle

from lib.planner import Planner
from lib.sensor import Sensor
from lib.occlusion_tracker import Occlusion_tracker
from lib.utilities import add_no_stop_zone
from lib.helper_functions import euler_from_quaternion

from datmo.msg import TrackArray
from nav_msgs.msg import Odometry


class ForeseeTheUnseenNode(Node):
    def __init__(self):
        super().__init__("foresee_the_unseen_node")

        # make parameters -- YAML file
        self.map_frame = "map"
        self.config_yaml = os.path.join(
            get_package_share_directory("foresee_the_unseen"), "config/commonroad_scenario.yaml"
        )
        self.road_xml = os.path.join(get_package_share_directory("foresee_the_unseen"), "resource/road_structure.xml")
        self.datmo_topic = "datmo/box_kf"
        self.odom_topic = "odom" # TODO: maybe use filtered odom
        self.motor_topic = None
        self.frequency = 2

        # subscriptions and publishers
        self.datmo_subscription = self.create_subscription(TrackArray, self.datmo_topic, self.datmo_callback, 5)
        self.odom_subscription = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 5)
        self.motor_publisher = None

        self.detected_obstacles = None
        self.ego_vehicle_state = None
        self.time_step = 0

        self.create_timer(1/self.frequency, self.update_scenario)

    def datmo_callback(self, msg: TrackArray):
        """Converts DATMO detections to Commonroad obstacles"""
        # TODO: Implement
        self.detected_obstacles = []
    
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

    def odom_callback(self, msg: Odometry):
        """Converts the odometry data to a Commonroad state."""
        self.ego_vehicle_state = self.state_from_odom(msg, self.time_step)
    
    def initialize_commonroad(self):
        with open(self.config_yaml) as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)

        self.scenario, _ = CommonRoadFileReader(self.road_xml).open()

        ego_vehicle_initial_state = InitialState(
            position=np.array([self.configuration.get("initial_state_x"), self.configuration.get("initial_state_y")]),
            orientation=self.configuration.get("initial_state_orientation"),
            velocity=self.configuration.get("initial_state_velocity"),
            time_step=0,
        )
        ego_shape = Rectangle(self.configuration.get("vehicle_length"), self.configuration.get("vehicle_width"))
        # ego_vehicle_initial_state = InitialState(position=np.array([0, 0]), orientation=0, velocity=0, time_step=0)
        self.ego_vehicle = DynamicObstacle(self.scenario.generate_object_id(), ObstacleType.CAR, ego_shape, ego_vehicle_initial_state)

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

    @staticmethod
    def update_ego_vehicle(ego_vehicle: DynamicObstacle, state: State):
        """Update the ego vehicle based on the state and not based on the predicted trajectory."""
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

    def update_scenario(self):
        """Gets called every n seconds"""
        # TODO: keep track of the time_step -- verify that it works
        # TODO: obtain the detected obstacles from a ROS topic
        if self.detected_obstacles is None or self.ego_vehicle_state is None:
            return
        
        percieved_scenario = copy.deepcopy(self.scenario)
        self.ego_vehicle = self.update_ego_vehicle(self.ego_vehicle, self.ego_vehicle_state)  # only update for visualization ... ??!!

        percieved_scenario.add_objects(self.detected_obstacles)

        # Update the sensor and get the sensor view and the list of observed obstacles
        self.sensor.update(self.ego_vehicle.initial_state)
        sensor_view = self.sensor.get_sensor_view(percieved_scenario)

        # Update the tracker with the new sensor view and get the prediction for the shadows
        self.occ_track.update(sensor_view, self.ego_vehicle.initial_state.time_step)
        shadow_obstacles = self.occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        add_no_stop_zone(
            percieved_scenario, self.time_step + self.configuration.get("planning_horizon"), self.configuration.get("safety_margin")
        )  # should not be necessary in every timestep
        self.planner.update(self.ego_vehicle.initial_state)
        collision_free_trajectory = self.planner.plan(percieved_scenario)
        if collision_free_trajectory:
            self.ego_vehicle.prediction = collision_free_trajectory
        # else, if no trajectory found, keep previous collision free trajectory
        
        # TODO: send motor commands based on trajectory

        # TODO: save to disk every timestep to be able to kill the process at any time
        # percieved_scenario.add_objects(self.ego_vehicle)
        # self.percieved_scenarios.append(percieved_scenario)
        # self.sensor_views.append(sensor_view)
        # self.driven_state_list.append(self.ego_vehicle.initial_state)

        # TODO: RVIZ visualization

        self.time_step += 1


def main(args=None):
    rclpy.init(args=args)

    foresee_the_unseen_node = ForeseeTheUnseenNode()

    rclpy.spin(foresee_the_unseen_node)
    foresee_the_unseen_node.destroy_node()
    rclpy.shutdown()
    