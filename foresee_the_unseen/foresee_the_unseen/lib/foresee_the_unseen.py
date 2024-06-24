# import warnings
# warnings.filterwarnings("error")
import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


# warnings.showwarning = warn_with_traceback

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
from foresee_the_unseen.lib.utilities import add_no_stop_zone, Logger, PrintLogger, get_no_stop_shape
from foresee_the_unseen.lib.helper_functions import (
    euler_from_quaternion,
    create_log_directory,
    recursive_getitem,
)

# to load the pickled error models
import foresee_the_unseen.lib.error_model as error_model
import sys

sys.modules["error_model"] = error_model


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
        logger: Logger = PrintLogger(),
        log_dir: Optional[str] = None,
        error_models_dir: Optional[str] = None,
    ):
        self.config_yaml = config_yaml
        self.road_xml = road_xml
        self.logger = logger
        self.frequency = frequency
        self.throttle_duration = 3  # set the throttle duration for the logging when used with ROS
        self.do_track_exec_time = True

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

        self.error_models_dir = error_models_dir
        if self.error_models_dir is not None:
            try:
                filepath = os.path.join(self.error_models_dir, "long_dt_error_model.pickle")
                with open(filepath, "rb") as f:
                    long_dt_error_model = pickle.load(f)
                long_dt_error_model.bounds_error = False
                self.logger_info(f"`Trajectory Longitudinal error rate model` loaded from {filepath} ")
                self.logger_info(str(long_dt_error_model))
            except FileNotFoundError:
                self.logger_warn(f"No `Trajectory Longitudinal error rate model` found at {filepath}")
                long_dt_error_model = None
            try:
                filepath = os.path.join(self.error_models_dir, "lat_error_model.pickle")
                with open(filepath, "rb") as f:
                    lat_error_model = pickle.load(f)
                lat_error_model.bounds_error = False
                self.logger_info(f"`Lateral error model` loaded from {filepath} ")
                self.logger_info(str(lat_error_model))
            except FileNotFoundError:
                self.logger_warn(f"No `Lateral error model` found at {filepath}")
                lat_error_model = None

            try:
                filepath = os.path.join(self.error_models_dir, "orient_error_model.pickle")
                with open(filepath, "rb") as f:
                    orient_error_model = pickle.load(f)
                orient_error_model.bounds_error = False
                self.logger_info(f"`Orientation error model` loaded from {filepath} ")
                self.logger_info(str(orient_error_model))
            except FileNotFoundError:
                self.logger_warn(f"No `Orientation error model` found at {filepath}")
                orient_error_model = None
        else:
            long_dt_error_model, lat_error_model, orient_error_model = None, None, None

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

        lanes_to_merge = recursive_getitem(
            self.configuration,
            ["lanes_to_track", self.scenario.scenario_id.map_name],
            default=None,
        )
        if lanes_to_merge is None:
            self.logger.warn("No specific lanes are selected for the occlusion tracker.")
        self.occ_track = Occlusion_tracker(
            scenario=self.scenario,
            min_vel=self.configuration["min_velocity"],
            max_vel=self.configuration["max_velocity"],
            min_shadow_area=self.configuration["min_shadow_area"],
            prediction_horizon=self.configuration["prediction_horizon"],
            steps_per_occ_pred=self.configuration["prediction_step_size"],
            dt=1 / self.frequency,
            tracking_enabled=self.configuration["tracking_enabled"],
            lanes_to_merge=lanes_to_merge,
        )

        no_stop_lanelet_id = recursive_getitem(
            self.configuration, ["no_stop_lanelets", self.scenario.scenario_id.map_name]
        )
        self.no_stop_shape = get_no_stop_shape(self.scenario, no_stop_lanelet_id, self.configuration["safety_margin"])
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
            apply_error_models=self.configuration["apply_error_models"],
            localization_position_std=recursive_getitem(
                self.configuration,
                ["localization_standard_deviations", "position"],
                default=None,
            ),
            localization_orientation_std=recursive_getitem(
                self.configuration,
                ["localization_standard_deviations", "orientation"],
                default=None,
            ),
            longitudinal_error_rate_model=long_dt_error_model,
            lateral_error_model=lat_error_model,
            orientation_error_model=orient_error_model,
            z_values_configuration=self.configuration.get("z_values_planner", None),
        )
        self.logger.info("commonroad scenario initialized")

        # Planner variables
        self.detected_obstacles: Optional[List[Obstacle]] = None
        self._ego_vehicle_state: Optional[InitialState] = None
        self._ego_vehicle_state_stamp: float = 0.0
        self.trajectory: Optional[Trajectory] = None
        self.planner_step: int = 0
        self._field_of_view: Optional[ShapelyPolygon] = None
        self._field_of_view_stamp: float = 0.0

    def logger_warn(self, message: str, **kwargs):
        if self.logger is not None:
            self.logger.warn(("[foresee-the-unseen] " + message), **kwargs)

    def logger_info(self, message: str, **kwargs):
        if self.logger is not None:
            self.logger.info(("[foresee-the-unseen] " + message), **kwargs)

    def position_on_road_check(self, x: float, y: float):
        """Check if a certain position is on the road"""

    def set_ego_vehicle_state(self, state: InitialState, time_stamp: float):
        """State should be in the planner frame, which is the Commonroad frame"""
        assert isinstance(state, InitialState), f"`state` should be of type {InitialState}, but is {type(state)}"
        self._ego_vehicle_state_stamp = time_stamp
        self._ego_vehicle_state = state

    def get_ego_vehicle_state(self, current_time: float) -> Optional[InitialState]:
        """Return the ego_vehicle state with some margin"""
        return self._ego_vehicle_state

    def set_field_of_view(self, field_of_view: ShapelyPolygon, time_stamp: float) -> None:
        """FOV should be in the planner frame, which is the Commonroad frame"""
        assert isinstance(
            field_of_view, ShapelyPolygon
        ), f"`field_of_view` should be of type {ShapelyPolygon}, but is {type(field_of_view)}"
        self._field_of_view_stamp = time_stamp
        self._field_of_view = field_of_view

    def get_field_of_view(self, current_time: float) -> Optional[ShapelyPolygon]:
        """Return the FOV with a padding"""
        if self._field_of_view is None:
            return None
        # padding = (current_time - self._field_of_view_stamp) * self.configuration["max_velocity"]
        # return self._field_of_view.buffer(-padding)  # type: ignore
        return self._field_of_view

    @staticmethod
    def update_vehicle(vehicle: DynamicObstacle, state: InitialState):
        """Update the ego vehicle based on the state"""
        return DynamicObstacle(
            obstacle_id=vehicle.obstacle_id,
            obstacle_type=vehicle.obstacle_type,
            obstacle_shape=vehicle.obstacle_shape,
            initial_state=state,
        )

    # def save_scenario_and_prediction(self, scenario: Scenario, prediction: SetBasedPrediction, success: bool) -> None:
    #     with open(f"/home/christiaan/thesis/thesis_code/debug_planner/data/run1/scenario_{self.planner_step}", "wb") as f:
    #         pickle.dump(scenario, f)
    #     with open(f"/home/christiaan/thesis/thesis_code/debug_planner/data/run1/prediction_{self.planner_step}", "wb") as f:
    #         pickle.dump(prediction, f)
    #     with open(f"/home/christiaan/thesis/thesis_code/debug_planner/data/run1/success_{self.planner_step}", "wb") as f:
    #         pickle.dump(success, f)

    def update_scenario(self, plan_start_time: float) -> Tuple[
        List[DynamicObstacle],
        ShapelyPolygon,
        Optional[Trajectory],
        DynamicObstacle,
        Optional[SetBasedPrediction],
        float,
    ]:
        """Gets called at a certain rate"""
        self.planner_step += 1

        if self.do_track_exec_time:
            execution_times = {}
            start_time = time.time()
            interm_time = time.time()

        ego_vehicle_state = self.get_ego_vehicle_state(plan_start_time)
        if ego_vehicle_state is None or (self._ego_vehicle_state_stamp + 1 / self.frequency) < plan_start_time:
            self.logger_warn(
                "No up-to-date ego vehicle state available. ", throttle_duration_sec=self.throttle_duration
            )
            raise NoUpdatePossible()
        ego_vehicle_state.time_step = self.planner_step

        percieved_scenario = copy.deepcopy(self.scenario)  # start with a clean copy of the scenario
        self.ego_vehicle = self.update_vehicle(self.ego_vehicle, ego_vehicle_state)
        # percieved_scenario.add_objects(self.get_obstacles(percieved_scenario))  # type: ignore

        if self.do_track_exec_time:
            execution_times["init + make scenario"] = time.time() - interm_time
            interm_time = time.time()

        # Update the sensor view:
        field_of_view = self.get_field_of_view(plan_start_time)  # Based on laser scan
        if field_of_view is None:
            self.logger_warn("No FOV available")
            raise NoUpdatePossible()

        if self.do_track_exec_time:
            execution_times["field_of_view"] = time.time() - interm_time
            interm_time = time.time()

        # Update the tracker with the new sensor view and get the shadows and their prediction
        scan_delay = plan_start_time - self._field_of_view_stamp if self.configuration["do_account_scan_delay"] else 0.0
        self.logger.info(f"scan delay = {scan_delay * 1000:.0f} ms")
        self.occ_track.update(field_of_view, self.planner_step, scan_delay)
        shadow_obstacles = self.occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        projected_occluded_area = self.occ_track.get_currently_occluded_area()

        # Assign an area where the vehicle cannot stop / is no safe state
        no_stop_zone_obstacle = add_no_stop_zone(
            percieved_scenario,
            self.planner_step + self.configuration.get("planning_horizon") - 1,
            self.no_stop_shape,
        )

        if self.do_track_exec_time:
            execution_times["shadows + no_stop"] = time.time() - interm_time
            interm_time = time.time()

        # Update the planner and plan a trajectory
        try:
            time_left = 1 / self.frequency - (time.time() - plan_start_time) - 40e-3  # 40 ms margin
            self.planner.update(self.ego_vehicle.initial_state)  # type: ignore
            trajectory, prediction = self.planner.plan(percieved_scenario, time_left, self.trajectory)
            self.ego_vehicle.prediction = prediction
            self.trajectory = copy.deepcopy(trajectory)
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
                "sensor_view": field_of_view,
                "trajectory": trajectory,
            }
            filename = os.path.join(self.log_dir, f"step {self.planner_step}.pickle")
            with open(filename, "wb") as handle:
                pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # self.logger.info(f"Scenario updated: planner step = {self.planner_step}")

        if self.do_track_exec_time:
            execution_times["logging"] = time.time() - interm_time
            execution_times["total"] = time.time() - start_time
            self.logger.info(str({key: round(value * 1000) for key, value in execution_times.items()}))

        if trajectory is not None:
            dt = 1 / self.frequency
            for state in trajectory.state_list:
                state.time_step = (state.time_step - self.planner_step) * dt + plan_start_time
            # self.logger.info(str([s.orientation for s in trajectory.state_list]) + "\n")

        return (
            shadow_obstacles,
            field_of_view,
            trajectory,
            no_stop_zone_obstacle,
            prediction,
            projected_occluded_area,
        )  # type: ignore
