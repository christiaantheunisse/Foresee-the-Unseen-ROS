import os
import yaml
import pickle
import copy
import numpy as np
from typing import Dict, Optional, Callable, List

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle, Obstacle
from commonroad.geometry.shape import Rectangle, Circle

from shapely.geometry import Polygon as ShapelyPolygon

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    matrices_from_cw_cvx_polygon,
)
from foresee_the_unseen.lib.planner import Planner, PositionNotOnALane, GoalAndPositionNotSameLane
from foresee_the_unseen.lib.sensor import Sensor
from foresee_the_unseen.lib.occlusion_tracker import Occlusion_tracker
from foresee_the_unseen.lib.utilities import add_no_stop_zone
from foresee_the_unseen.lib.helper_functions import euler_from_quaternion, create_log_directory


class NoUpdatePossible(Exception):
    """The position is not on a lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class Logger:
    def info(message: str, **kwargs):
        return

    def warn(message: str, **kwargs):
        return


class PrintLogger(Logger):
    def info(message: str, **kwargs):
        print("[INFO]", message)

    def warn(message: str, **kwargs):
        print("[WARN]", message)


class ForeseeTheUnseen:
    """
    This class implements the algorithm for the trajectory planning from: Foresee the Unseen: Sequential Reasoning
    about Hidden Obstacles for Safe Driving (Sanchez et al., 2022).

    This class is ROS agnostic.
    """

    def __init__(self, config_yaml: str, road_xml: str, frequency: float, logger=None, log_dir=None):
        self.config_yaml = config_yaml
        self.road_xml = road_xml
        self.logger = logger if logger is not None else Logger()
        self.frequency = frequency
        self.throttle_duration = 2  # set the throttle duration for the logging when used with ROS

        if self.logger is not None:
            assert hasattr(self.logger, "info") and hasattr(
                self.logger, "warn"
            ), "The logger should have methods `info` and `warn`"
            assert callable(self.logger.info) and callable(
                self.logger.warn
            ), "The methods `info` and `warn` of the logger should be callable"

        # Read the files
        with open(self.config_yaml) as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)
        self.scenario, _ = CommonRoadFileReader(self.road_xml).open()

        # Make directory for logging
        self.log_dir = create_log_directory(log_dir) if log_dir is not None else None

        # Save the configuration to disk for logging purposes
        if self.log_dir is not None:
            settings_dict = self.configuration
            settings_dict["road_xml"] = self.road_xml
            settings_dict["frequency"] = self.frequency
            filename = os.path.join(self.log_dir, "settings.pickle")
            with open(filename, "wb") as handle:
                pickle.dump(settings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Setup the objects necessary for the planner
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
        self.logger.info("commonroad scenario initialized")

        # Planner variables
        self.detected_obstacles = None
        self.ego_vehicle_state = None
        self.trajectory = None
        self.planner_step = 0
        self.sensor_view = None

    def position_on_road_check(self, x: float, y: float):
        """Check if a certain position is on the road"""


    def update_state(self, state: InitialState):
        """State should be in the planner frame, which is the Commonroad frame"""
        state.time_step = self.planner_step
        self.ego_vehicle_state = state

    def update_fov(self, fov: ShapelyPolygon):
        """FOV should be in the planner frame, which is the Commonroad frame"""
        self.sensor_view = fov

    def update_obstacles(self, detected_obstacles: List[Obstacle]):
        """Obstacles should be in the planner frame, which is the Commonroad frame"""
        self.detected_obstacles = detected_obstacles

    @staticmethod
    def update_vehicle(vehicle: DynamicObstacle, state: State):
        """Update the ego vehicle based on the state"""
        return DynamicObstacle(
            obstacle_id=vehicle.obstacle_id,
            obstacle_type=vehicle.obstacle_type,
            obstacle_shape=vehicle.obstacle_shape,
            initial_state=state,
        )

    def update_scenario(self):
        """Gets called at a certain rate"""
        no_obstacles = self.detected_obstacles is None
        no_updated_state = self.ego_vehicle_state is None or self.ego_vehicle_state.time_step < self.planner_step
        if no_obstacles or no_updated_state:
            if no_obstacles:
                self.logger.warn("No detected obstacles available")
            if no_updated_state:
                self.logger.warn("No up-to-date ego vehicle state available")
            raise NoUpdatePossible()

        percieved_scenario = copy.deepcopy(self.scenario)  # start with a clean scenario
        self.ego_vehicle = self.update_vehicle(self.ego_vehicle, self.ego_vehicle_state)  # update ego vehicle
        percieved_scenario.add_objects(self.detected_obstacles)  # a

        # Update the sensor view:
        if self.configuration.get("laser_scan_fov"):
            sensor_view = self.sensor_view  # Based on laser scan
            if sensor_view is None:
                self.logger.warn("No FOV available", throttle_duration_sec=self.throttle_duration)
        else:
            self.sensor.update(self.ego_vehicle.initial_state)
            sensor_view = self.sensor.get_sensor_view(percieved_scenario)  # Based on detected obstacles

        # Update the tracker with the new sensor view and get the shadows and their prediction
        self.occ_track.update(sensor_view, self.ego_vehicle.initial_state.time_step)
        shadow_obstacles = self.occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        add_no_stop_zone(
            percieved_scenario,
            self.planner_step + self.configuration.get("planning_horizon"),
            self.configuration.get("safety_margin"),
        )  # should not be necessary in every timestep

        try:
            self.planner.update(self.ego_vehicle.initial_state)  # FIXME: should not be possible to remove all waypoints
        # FIXME: Throws an attribute error if the ego_vehicle or goal_position is not on the road
            collision_free_trajectory = self.planner.plan(percieved_scenario)
        except PositionNotOnALane as p:
            self.logger.warn(f"Position not on a lane: {p.message}")
            raise NoUpdatePossible()
        except GoalAndPositionNotSameLane as g:
            self.logger.warn(f"Goal and position not on the same lane: {g.message}")
            raise NoUpdatePossible()
        
        if collision_free_trajectory:
            self.logger.info("new trajectory found")
            self.trajectory = collision_free_trajectory

        percieved_scenario.add_objects(self.ego_vehicle)

        if self.log_dir is not None:
            log_dict = {"ego_vehicle": self.ego_vehicle, "scenario": percieved_scenario, "sensor_view": sensor_view}
            filename = os.path.join(self.log_dir, f"step {self.planner_step}.pickle")
            with open(filename, "wb") as handle:
                pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.planner_step += 1
        self.logger.info(f"Scenario updated: planner step = {self.planner_step}")

        return percieved_scenario, sensor_view
