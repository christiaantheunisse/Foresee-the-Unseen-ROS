import os
import yaml
import pickle
import copy
import time
import numpy as np
from typing import Dict, Optional, Callable, List, Type, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import SetBasedPrediction
from commonroad.scenario.obstacle import (
    DynamicObstacle,
    ObstacleType,
    StaticObstacle,
    Obstacle,
)
from commonroad.geometry.shape import Rectangle, Circle

from shapely.geometry import Polygon as ShapelyPolygon

from foresee_the_unseen.lib.helper_functions import (
    polygons_from_road_xml,
    matrix_from_transform,
    matrices_from_cw_cvx_polygon,
)
from foresee_the_unseen.lib.planner import (
    Planner,
    PositionNotOnALane,
    GoalAndPositionNotSameLane,
    NoSafeTrajectoryFound,
)
from foresee_the_unseen.lib.sensor import Sensor
from foresee_the_unseen.lib.occlusion_tracker import Occlusion_tracker
from foresee_the_unseen.lib.utilities import add_no_stop_zone, Logger, PrintLogger
from foresee_the_unseen.lib.helper_functions import (
    euler_from_quaternion,
    create_log_directory,
)


class NoUpdatePossible(Exception):
    """The position is not on a lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class ForeseeTheUnseen:
    """
    This class implements the algorithm for the trajectory planning from: Foresee the Unseen: Sequential Reasoning
    about Hidden Obstacles for Safe Driving (Sanchez et al., 2022).

    This class is ROS agnostic.
    """

    def __init__(
        self,
        config_yaml: str,
        road_xml: str,
        frequency: float,
        logger: Type[Logger] = PrintLogger,
        log_dir: Optional[str] = None,
    ):
        self.config_yaml = config_yaml
        self.road_xml = road_xml
        self.logger = logger
        self.frequency = frequency
        self.throttle_duration = 3  # set the throttle duration for the logging when used with ROS
        self.do_track_exec_time = False

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
        ego_shape = Rectangle(
            self.configuration["vehicle_length"],
            self.configuration["vehicle_width"],
        )
        ego_vehicle_initial_state = InitialState(position=np.array([0, 0]), orientation=0, velocity=0, time_step=0)
        self.ego_vehicle = DynamicObstacle(
            self.scenario.generate_object_id(),
            ObstacleType.CAR,
            ego_shape,
            ego_vehicle_initial_state,
        )

        self.sensor = Sensor(
            position=self.ego_vehicle.initial_state.position,
            orientation=self.ego_vehicle.initial_state.orientation,
            field_of_view=self.configuration["field_of_view_degrees"] * 2 * np.pi / 360,
            min_resolution=self.configuration["min_resolution"],
            view_range=self.configuration["view_range"],
        )

        self.occ_track = Occlusion_tracker(
            scenario=self.scenario,
            min_vel=self.configuration["min_velocity"],
            max_vel=self.configuration["max_velocity"],
            min_shadow_area=self.configuration["min_shadow_area"],
            prediction_horizon=self.configuration["prediction_horizon"],
            steps_per_occ_pred=self.configuration["prediction_step_size"],
            dt=1 / self.frequency,
            tracking_enabled=self.configuration["tracking_enabled"],
        )

        self.planner = Planner(
            lanelet_network=self.scenario.lanelet_network,
            initial_state=self.ego_vehicle.initial_state,
            goal_point=[
                self.configuration["goal_point_x"],
                self.configuration["goal_point_y"],
            ],
            vehicle_size=(
                self.configuration["vehicle_length"],
                self.configuration["vehicle_width"],
            ),
            vehicle_center_offset=(
                self.configuration["vehicle_length_offset"],
                self.configuration["vehicle_width_offset"],
            ),
            reference_speed=self.configuration["reference_speed"],
            max_acceleration=self.configuration["max_acceleration"],
            max_deceleration=self.configuration["max_deceleration"],
            time_horizon=self.configuration["planning_horizon"],
            dt=1 / self.frequency,
            min_dist_waypoint=self.configuration["minimum_distance_waypoint"],
            logger=self.logger,
            max_dist_corner_smoothing=self.configuration["max_dist_corner_smoothing"],
        )
        self.logger.info("commonroad scenario initialized")

        # Planner variables
        self.detected_obstacles: Optional[List[Obstacle]] = None
        self.ego_vehicle_state: Optional[InitialState] = None
        self.trajectory: Optional[Trajectory] = None
        self.planner_step: int = 0
        self.sensor_view: Optional[ShapelyPolygon] = None

    def logger_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(
                ("[foresee-the-unseen] " + message),
                throttle_duration_sec=self.throttle_duration,
            )

    def logger_info(self, message: str):
        if self.logger is not None:
            self.logger.info(
                ("[foresee-the-unseen] " + message),
                throttle_duration_sec=self.throttle_duration,
            )

    def position_on_road_check(self, x: float, y: float):
        """Check if a certain position is on the road"""

    def update_state(self, state: InitialState):
        """State should be in the planner frame, which is the Commonroad frame"""
        state.time_step = self.planner_step
        self.ego_vehicle_state = state

    def update_fov(self, fov: ShapelyPolygon):
        """FOV should be in the planner frame, which is the Commonroad frame"""
        self.sensor_view = fov

    # Make a property
    def update_obstacles(self, detected_obstacles: List[Obstacle]):
        """Obstacles should be in the planner frame, which is the Commonroad frame"""
        self.detected_obstacles = detected_obstacles

    def get_obstacles(self, scenario: Scenario) -> List[Obstacle]:
        for obs in self.detected_obstacles:
            obs.initial_state.time_step = self.planner_step
            # obstacle_id is an immutable property; this overrides the constraint
            obs._obstacle_id = scenario.generate_object_id()
        return self.detected_obstacles

    @staticmethod
    def update_vehicle(vehicle: DynamicObstacle, state: InitialState):
        """Update the ego vehicle based on the state"""
        return DynamicObstacle(
            obstacle_id=vehicle.obstacle_id,
            obstacle_type=vehicle.obstacle_type,
            obstacle_shape=vehicle.obstacle_shape,
            initial_state=state,
        )

    def update_scenario(
        self,
    ) -> Tuple[
        List[DynamicObstacle],
        ShapelyPolygon,
        Optional[Trajectory],
        ShapelyPolygon,
        Optional[SetBasedPrediction],
    ]:
        """Gets called at a certain rate"""
        if self.do_track_exec_time:
            execution_times = {}
            start_time = time.time()
            interm_time = time.time()

        # Do not require detected obstacles
        # TODO: Remove detected obstacles
        self.detected_obstacles = [] if self.detected_obstacles is None else self.detected_obstacles
        no_obstacles = self.detected_obstacles is None
        no_updated_state = self.ego_vehicle_state is None or self.ego_vehicle_state.time_step < self.planner_step
        if no_obstacles or no_updated_state:
            if no_obstacles:
                self.logger_warn("No detected obstacles available")
            if no_updated_state:
                self.logger_warn("No up-to-date ego vehicle state available")
            raise NoUpdatePossible()

        percieved_scenario = copy.deepcopy(self.scenario)  # start with a clean scenario
        self.ego_vehicle = self.update_vehicle(self.ego_vehicle, self.ego_vehicle_state)
        percieved_scenario.add_objects(self.get_obstacles(percieved_scenario))  # type: ignore

        if self.do_track_exec_time:
            execution_times["init + make scenario"] = time.time() - interm_time
            interm_time = time.time()

        # Update the sensor view:
        if self.configuration.get("laser_scan_fov", False):
            sensor_view = self.sensor_view  # Based on laser scan
            if sensor_view is None:
                self.logger_warn("No FOV available")
                raise NoUpdatePossible()
        else:
            self.sensor.update(self.ego_vehicle.initial_state)
            sensor_view = self.sensor.get_sensor_view(percieved_scenario)  # Based on detected obstacles

        if self.do_track_exec_time:
            execution_times["sensor_view"] = time.time() - interm_time
            interm_time = time.time()

        # Update the tracker with the new sensor view and get the shadows and their prediction
        self.occ_track.update(sensor_view, self.ego_vehicle.initial_state.time_step)
        shadow_obstacles = self.occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        no_stop_zone_obstacle = add_no_stop_zone(
            percieved_scenario,
            self.planner_step + self.configuration.get("planning_horizon") - 1,
            self.configuration.get("safety_margin"),
        )  # should not be necessary in every timestep

        if self.do_track_exec_time:
            execution_times["shadows + no_stop"] = time.time() - interm_time
            interm_time = time.time()

        try:
            self.planner.update(self.ego_vehicle.initial_state)  # type: ignore
            trajectory, prediction = self.planner.plan(percieved_scenario)
            self.ego_vehicle.prediction = prediction
            # self.trajectory = trajectory
        except NoSafeTrajectoryFound:
            self.logger_info("No safe trajectory found")
            trajectory, prediction = None, None
        except PositionNotOnALane as p:
            trajectory, prediction = None, None
            self.logger_warn(f"Position not on a lane: {p.message}")
            raise NoUpdatePossible()
        except GoalAndPositionNotSameLane as g:
            trajectory, prediction = None, None
            self.logger_warn(f"Goal and position not on the same lane: {g.message}")
            raise NoUpdatePossible()

        if self.do_track_exec_time:
            execution_times["planner"] = time.time() - interm_time
            interm_time = time.time()

        percieved_scenario.add_objects(self.ego_vehicle)

        if self.log_dir is not None:
            log_dict = {
                "ego_vehicle": self.ego_vehicle,  # with the set_based_prediction
                "scenario": percieved_scenario,
                "sensor_view": sensor_view,
                "trajectory": trajectory,
            }
            filename = os.path.join(self.log_dir, f"step {self.planner_step}.pickle")
            with open(filename, "wb") as handle:
                pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.planner_step += 1
        # self.logger.info(f"Scenario updated: planner step = {self.planner_step}")

        if self.do_track_exec_time:
            execution_times["logging"] = time.time() - interm_time
            execution_times["total"] = time.time() - start_time
            self.logger.info(str({key: (value * 1000) for key, value in execution_times.items()}))

        return (
            shadow_obstacles,
            sensor_view,
            trajectory,
            no_stop_zone_obstacle,
            prediction,
        )  # type: ignore
