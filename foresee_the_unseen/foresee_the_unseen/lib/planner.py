import numpy as np
import math
from typing import Optional, List, Union, Tuple, Any, TypedDict, Type
import nptyping as npt
import multiprocessing

from commonroad.scenario.scenario import Scenario, LaneletNetwork
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState, State
from commonroad.prediction.prediction import (
    TrajectoryPrediction,
    SetBasedPrediction,
    Occupancy,
)
from commonroad.geometry.shape import Polygon
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
from commonroad.common.util import Interval

from foresee_the_unseen.lib.utilities import Lanelet2ShapelyPolygon, Logger, PrintLogger
from shapely.geometry import Point
from shapely.geometry import LineString

Scalar = Union[float, int]
PlanningResult = TypedDict(
    "PlanningResult", {"trajectory": Optional[Trajectory], "prediction": Optional[SetBasedPrediction]}
)


class PositionNotOnALane(Exception):
    """The position is not on a lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class GoalAndPositionNotSameLane(Exception):
    """The goal and the position are not on the same lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class NoSafeTrajectoryFound(Exception): ...


class FlatOccupancy(Exception): ...


class Planner:
    def __init__(
        self,
        lanelet_network: LaneletNetwork,
        initial_state,
        goal_point,
        vehicle_size: Tuple[float, float],
        vehicle_center_offset: Tuple[float, float],
        reference_speed,
        max_acceleration,
        max_deceleration,
        time_horizon,
        dt,
        min_dist_waypoint,
        logger: Type[Logger] = PrintLogger,
        max_dist_corner_smoothing: float = 0.1,
        waypoints: Optional[np.ndarray] = None,
    ):
        self.lanelet_network = lanelet_network
        self.initial_state = initial_state
        self.goal_point = goal_point
        self.vehicle_size = vehicle_size
        self.vehicle_center_offset = vehicle_center_offset
        self.reference_speed = reference_speed
        self.max_acc = max_acceleration
        self.max_dec = np.abs(max_deceleration)
        self.time_horizon = time_horizon
        self.dt = dt
        self.min_dist_waypoint = min_dist_waypoint
        self.logger = logger
        if waypoints is not None:
            self.waypoints = waypoints
        else:
            self._waypoints = None
            self._passed_waypoints = 0
        self.max_dist_corner_smoothing = max_dist_corner_smoothing

        self.goal_reached = False

    @property
    def waypoints(self) -> npt.NDArray[npt.Shape["N, 2"], npt.Float]:
        return self._waypoints  # type: ignore

    @waypoints.setter
    def waypoints(self, value: npt.NDArray[npt.Shape["N, 2"], npt.Float]) -> None:
        self._passed_waypoints = 0
        self._waypoints = value
        self._clear_waypoints_variables()
        self.update_passed_waypoints()

    @property
    def waypoints_w_ego_pos(self) -> npt.NDArray[npt.Shape["N, 2"], npt.Float]:
        if self._waypoints_w_ego_pos is None:
            self._set_waypoints_variables()
        return self._waypoints_w_ego_pos  # type: ignore

    @property
    def passed_waypoints(self) -> int:
        return self._passed_waypoints

    @passed_waypoints.setter
    def passed_waypoints(self, value: int) -> None:
        self._clear_waypoints_variables()
        if value > self.goal_waypoint_idx:
            self.goal_reached = True
        self._passed_waypoints = min(value, self.goal_waypoint_idx)  # type: ignore

    @property
    def dist_along_points(self) -> npt.NDArray[npt.Shape["N"], npt.Float]:
        """
        The cumulative distance along the waypoints, where...
            ...current position/waypoint is 0
            ...position ahead > 0
            ...position behind < 0
        """
        if self._dist_along_points is None:
            self._set_waypoints_variables()
        return self._dist_along_points  # type: ignore

    @property
    def orient_bw_points(self) -> npt.NDArray[npt.Shape["N"], npt.Float]:
        """The counter clockwise orientation between the waypoints in rad. Along x-axis = 0."""
        if self._orient_bw_points is None:
            self._set_waypoints_variables()
        return self._orient_bw_points  # type: ignore

    def _set_waypoints_variables(self) -> None:
        self._waypoints_w_ego_pos = np.insert(
            self.waypoints, self.passed_waypoints, self.initial_state.position, axis=0
        )
        diff_points = np.diff(self._waypoints_w_ego_pos, axis=0)
        dist_bw_points = np.hypot(*diff_points.T)
        self._dist_along_points = np.hstack(([0], np.cumsum(dist_bw_points)))
        self._dist_along_points -= self._dist_along_points[self.passed_waypoints]
        self._orient_bw_points = np.arctan2(*diff_points.T[::-1])

    def _clear_waypoints_variables(self) -> None:
        self._waypoints_w_ego_pos = None
        self._dist_along_points = None
        self._orient_bw_points = None

    def update(self, state: InitialState) -> None:
        if isinstance(state.velocity, Scalar):
            state.velocity = max(state.velocity, 0.0)
        elif isinstance(state.velocity, Interval):
            state.velocity.start = max(state.velocity.start, 0.0)
            state.velocity.end = max(state.velocity.end, 0.0)
        else:
            raise TypeError
        self.initial_state = state
        if self.waypoints is None:
            self.find_waypoints()
        self._clear_waypoints_variables()
        self.update_passed_waypoints()

    def plan(self, scenario: Scenario) -> Tuple[Trajectory, SetBasedPrediction]:
        """Finds the fastest possible trajectory by iterative trying different velocity profiles."""
        # TODO: make comp time dependant
        if self.goal_reached:
            raise NoSafeTrajectoryFound

        best_result_so_far: PlanningResult = {"trajectory": None, "prediction": None}

        loc_pos_error = 0.0
        loc_orient_error = 0.0
        traj_long_error = 0.0
        traj_lat_error = 0.0
        traj_orient_error = 0.0

        velocity_profiles = self.generate_velocity_profiles()
        self.collision_checker = create_collision_checker(scenario)

        def check_velocity_profile(idx: int) -> bool:
            velocity_profile = velocity_profiles[idx]
            set_based_prediction = self.create_occupancy_set(
                velocity_profile=velocity_profile,
                loc_pos_error=loc_pos_error,
                loc_orient_error=loc_orient_error,
                traj_long_error=traj_long_error,
                traj_lat_error=traj_lat_error,
                traj_orient_error=traj_orient_error,
            )
            is_safe = self.is_safe_trajectory(set_based_prediction)
            if is_safe:
                best_result_so_far["trajectory"] = self.velocity_profile_to_state_list(velocity_profile)
                best_result_so_far["prediction"] = set_based_prediction

            return is_safe

        # If the fastest velocity profiles -> try the others
        if not check_velocity_profile(0):
            # Try slowest velocity profile and subsequently the others by a branch-and-bound approach starting at the center
            check_velocity_profile(-1)

            N = len(velocity_profiles)
            # TODO: implement some time limit here
            step_N, current_N = N / 2, N / 2
            while step_N >= 2 and -1e-6 < current_N < N + 1e-6:
                step_N /= 2
                current_N -= step_N if check_velocity_profile(round(current_N) - 1) else -step_N

        if best_result_so_far["prediction"] is not None and best_result_so_far["trajectory"]:
            return best_result_so_far["trajectory"], best_result_so_far["prediction"]
        else:
            raise NoSafeTrajectoryFound

    def velocity_profile_to_state_list(self, velocity_profile):
        if isinstance(self.initial_state.velocity, Scalar):
            assert velocity_profile.ndim == 1
            velocity_profile_avg = velocity_profile
            v_diffs = velocity_profile - np.insert(velocity_profile[:-1], 0, self.initial_state.velocity)
            accelerations = v_diffs / self.dt
        elif isinstance(self.initial_state.velocity, Interval):
            assert velocity_profile.ndim == 2
            init_velocity = (self.initial_state.velocity.start + self.initial_state.velocity.end) / 2
            velocity_profile_avg = velocity_profile.mean(axis=1)
            v_diffs = velocity_profile_avg - np.insert(velocity_profile_avg[:-1], 0, init_velocity)
            accelerations = v_diffs / self.dt

        # use the 'trapezoidal rule' for integrating the velocity
        # TODO: trapezoidal integration rule
        # velocity_extend = np.insert(velocity_profile, 0, self.initial_state.velocity)
        # velocity_trapez = (velocity_extend[:-1] + velocity_extend[1:]) / 2
        # dist = np.cumsum(velocity_trapez * self.dt)
        dist = np.cumsum(velocity_profile_avg * self.dt)
        x_coords = np.interp(dist, self.dist_along_points, self.waypoints_w_ego_pos[:, 0])
        y_coords = np.interp(dist, self.dist_along_points, self.waypoints_w_ego_pos[:, 1])

        # The trajectory orientation on the parts between the waypoints is constant, but the trajectory  orientation
        #  around the waypoints is smoothed. The smoothening happens within a distance of the minimum of
        #  (`self.max_dist_corner_smoothing`) and half of the length of the part between two waypoints.
        dist_along_bef_points = self.dist_along_points[1 + self.passed_waypoints : -1] - np.minimum(
            np.diff(self.dist_along_points[self.passed_waypoints : -1]) / 2,
            self.max_dist_corner_smoothing,
        )
        dist_along_aft_points = self.dist_along_points[1 + self.passed_waypoints : -1] + np.minimum(
            np.diff(self.dist_along_points[self.passed_waypoints + 1 :]) / 2,
            self.max_dist_corner_smoothing,
        )
        dist_along_bef_aft_points = np.vstack((dist_along_bef_points, dist_along_aft_points)).T.flatten()
        orientations_at_points = np.repeat(self.orient_bw_points[self.passed_waypoints :], 2)[1:-1]
        orientations = np.interp(dist, dist_along_bef_aft_points, orientations_at_points)

        state_list = []
        for time_step, (
            x_coord,
            y_coord,
            orientation,
            velocity,
            acceleration,
        ) in enumerate(zip(x_coords, y_coords, orientations, velocity_profile_avg, accelerations, strict=True)):
            state = InitialState(
                position=np.array([x_coord, y_coord]),
                orientation=orientation,
                velocity=velocity,
                acceleration=acceleration,
                time_step=time_step + self.initial_state.time_step + 1,  # type: ignore
            )
            state_list.append(state)

        return Trajectory(self.initial_state.time_step + 1, state_list)  # type: ignore

    def find_waypoints(self):
        starting_lanelet_ids = self.lanelet_network.find_lanelet_by_position([self.initial_state.position])[0]  # type: ignore
        if starting_lanelet_ids == []:
            raise PositionNotOnALane(f"Current position: {self.initial_state.position}")
        starting_lanelets = []
        for lanelet_id in starting_lanelet_ids:
            starting_lanelets.append(self.lanelet_network.find_lanelet_by_id(lanelet_id))

        starting_lane = None
        for lanelet in starting_lanelets:
            starting_lanes = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, self.lanelet_network)[0]
            for lane in starting_lanes:
                lane_shape = Lanelet2ShapelyPolygon(lane)
                if lane_shape.intersects(Point(*self.goal_point)):
                    starting_lane = lane
                    break
            else:
                continue
            break
        if starting_lane is None:
            raise GoalAndPositionNotSameLane(
                f"Goal: {self.goal_point}; current position: {self.initial_state.position}"
            )

        # find waypoints beyond goal position.
        center_vertices = np.array(starting_lane.center_vertices)
        center_line_shapely = LineString(center_vertices)
        goal_shapely = Point(self.goal_point)
        dist_along_line = center_line_shapely.project(goal_shapely)
        dists = np.cumsum(np.linalg.norm(center_vertices[1:] - center_vertices[:-1], axis=1))
        assert dist_along_line < dists[-1], f"Goal point lies beyond the last waypoint: goal_point={self.goal_point}"
        idx_before = np.argmax(dist_along_line < dists)
        self.goal_waypoint_idx = idx_before + 1

        self.waypoints = np.vstack(
            (
                center_vertices[: idx_before + 1],
                [self.goal_point],
                center_vertices[idx_before + 1 :],
            )
        )

    def update_passed_waypoints(self) -> None:
        for waypoint in self.waypoints[self.passed_waypoints :]:
            direction_vector = waypoint - self.initial_state.position
            angle_to_next_point = np.arctan2(*np.flip(direction_vector))
            angle_diff_to_next_point = abs(
                (np.pi + angle_to_next_point - self.initial_state.orientation) % (2 * np.pi) - np.pi
            )
            next_point_is_too_close = bool(np.hypot(*direction_vector) < self.min_dist_waypoint)
            if next_point_is_too_close or (angle_diff_to_next_point > np.pi / 2):
                self.passed_waypoints += 1
            else:
                return

    @staticmethod
    def _generate_scalar_velocity_profiles(
        velocity: float,
        max_dec: float,
        max_acc: float,
        dt: float,
        time_horizon: int,
        reference_speed: float,
        number_of_trajectories: int,
        dist_to_goal: float,
    ):
        assert velocity >= 0.0, f"velocity should be bigger than zero: {velocity=}"
        velocity_decs = velocity - max_dec * dt * np.arange(time_horizon)

        # fastest possible velocity profile which is set below depending on the current velocity and the distance to
        #  the goal
        acc_profile = np.full(time_horizon, -max_dec)

        velocity = min(velocity, reference_speed) # FIXME: is mistake

        if velocity <= reference_speed:
            # maximum reachable velocity within the limited distance
            v_reachable = np.sqrt(
                2 * (dist_to_goal + velocity**2 / (2 * max_acc)) * max_acc * max_dec / (max_acc + max_dec)
            )

            if v_reachable < reference_speed:
                #                       <- reference velocity (not reached)
                # v-profile:   /\
                #                \
                t_increasing = (v_reachable - velocity) / max_acc
                num_incr_steps = t_increasing / dt

                acc_profile[: int(num_incr_steps)] = max_acc
                acc_profile[int(num_incr_steps)] = max_acc * (num_incr_steps % 1) + max_dec * (num_incr_steps % -1)
            else:
                #              .--.     <- reference velocity (reached)
                # v-profile:  /    \
                #                   \
                t_increasing = (reference_speed - velocity) / max_acc
                num_incr_steps = t_increasing / dt
                t_horizontal = (
                    dist_to_goal
                    - (reference_speed**2 - velocity**2) / (2 * max_acc)
                    - reference_speed**2 / (2 * max_dec)
                ) / reference_speed
                num_horz_steps = t_horizontal / dt
                t_decreasing = reference_speed / max_dec
                num_decr_steps = t_decreasing / dt

                acc_profile[: int(num_incr_steps)] = max_acc  # increasing velocity
                acc_profile[int(num_incr_steps)] = (
                    num_incr_steps % 1 * max_acc
                )  # smaller acc to just reach reference speed
                acc_profile[int(num_incr_steps) + 1 : int(num_incr_steps + num_horz_steps)] = 0  # constant velocity
                acc_profile[-int(num_decr_steps) - 1 :] = -max_dec  # ensure that decelerating part is long enough
        elif velocity > reference_speed:  # the current velocity is above the reference velocity
            #             \                 #             \
            # v-profile:   '--.     OR      # v-profile:   \      <- reference velocity
            #                  \            #               \
            t_decreasing = (velocity - reference_speed) / max_dec
            num_decr_steps = t_decreasing / dt
            t_horizontal = max((dist_to_goal - velocity**2 / (2 * max_dec)) / reference_speed, 0)
            num_horz_steps = t_horizontal / dt

            acc_profile[: int(num_decr_steps)] = -max_dec
            acc_profile[int(num_decr_steps)] = num_decr_steps % 1 * -max_dec
            acc_profile[int(num_decr_steps) + 1 : int(num_decr_steps + num_horz_steps)] = 0

        velocity_incs = velocity + np.cumsum(acc_profile * dt)
        max_velocity_increase = np.max(velocity_incs - velocity_decs)

        if max_velocity_increase <= 0:
            max_velocity_increase = 0.0
            number_of_trajectories = 1
            velocity_decs = velocity_incs
        assert velocity_decs[-1] <= 0

        velocity_profiles = []
        for velocity_increase in np.linspace(max_velocity_increase, 0, number_of_trajectories):
            unbounded_velocity_profile = np.minimum(velocity_incs, velocity_increase + velocity_decs)
            velocity_profile = np.maximum(unbounded_velocity_profile, 0)
            velocity_profiles.append(velocity_profile)

        return np.array(velocity_profiles)

    def generate_velocity_profiles(self, number_of_trajectories=10):
        dist_to_goal = self.dist_along_points[self.goal_waypoint_idx + 1]
        if isinstance(self.initial_state.velocity, Scalar):
            velocity_profiles = self._generate_scalar_velocity_profiles(
                velocity=self.initial_state.velocity,
                max_dec=self.max_dec,
                max_acc=self.max_acc,
                dt=self.dt,
                time_horizon=self.time_horizon,
                reference_speed=self.reference_speed,
                number_of_trajectories=number_of_trajectories,
                dist_to_goal=dist_to_goal,
            )
        elif isinstance(self.initial_state.velocity, Interval):
            velocity_profiles_max = self._generate_scalar_velocity_profiles(
                velocity=self.initial_state.velocity.end,
                max_dec=self.max_dec,
                max_acc=self.max_acc,
                dt=self.dt,
                time_horizon=self.time_horizon,
                reference_speed=self.reference_speed,
                number_of_trajectories=number_of_trajectories,
                dist_to_goal=dist_to_goal,
            )
            velocity_profiles_max = velocity_profiles_max[..., np.newaxis]
            v_range = self.initial_state.velocity.end - self.initial_state.velocity.start
            velocity_profiles_min = np.maximum(velocity_profiles_max - v_range, 0)
            velocity_profiles = np.concatenate((velocity_profiles_min, velocity_profiles_max), axis=2)
        else:
            raise TypeError

        return velocity_profiles

    def get_xy_ths_for_range(
        self,
        start_dist_time,
        end_dist_time,
        dist_along_points,
        orient_bw_points,
        waypoints,
    ):
        """Gives points on the path with the orientation of the path for each range of distance along the path."""
        # Get the x,y-coordinates for the start and end of the distance range
        x_start = np.interp(start_dist_time, dist_along_points, waypoints[:, 0])
        y_start = np.interp(start_dist_time, dist_along_points, waypoints[:, 1])
        x_end = np.interp(end_dist_time, dist_along_points, waypoints[:, 0])
        y_end = np.interp(end_dist_time, dist_along_points, waypoints[:, 1])

        # Get the indices of the intermediate points within the distance range
        idcs_interm_start = np.argmax(
            start_dist_time <= np.tile(dist_along_points.reshape(-1, 1), (1, len(start_dist_time))),
            axis=0,
        )
        idcs_interm_end = np.argmax(
            end_dist_time <= np.tile(dist_along_points.reshape(-1, 1), (1, len(end_dist_time))),
            axis=0,
        )
        th_start, th_end = (
            orient_bw_points[idcs_interm_start - 1],
            orient_bw_points[idcs_interm_end - 1],
        )
        idcs_in_range = np.vstack((idcs_interm_start, idcs_interm_end)).T
        assert not 0 in idcs_in_range, "The first waypoint lies within the predicted occupancy."
        # return an array with 3d vectors consisting of x-coordinate, y-coordinate and yaw angle (theta (th)) for all the
        #  points within the distance range where the orientation changes.
        xyth_start, xyth_end = (
            np.vstack((x_start, y_start, th_start)).T,
            np.vstack((x_end, y_end, th_end)).T,
        )
        for start_point, end_point, point_idcs in zip(xyth_start, xyth_end, idcs_in_range):
            points_in_range = np.hstack(
                (
                    waypoints[slice(*point_idcs)],
                    orient_bw_points[slice(*(point_idcs - 1)), np.newaxis],
                )
            )
            yield np.vstack(([start_point], points_in_range, [end_point]))

    def xy_ths_to_polygon(
        self,
        points_with_orient: npt.NDArray[npt.Shape["N, 3"], npt.Float],
        left_width: float,
        right_width: float,
    ) -> Polygon:
        """Converts the points on the path with an orientation to a polygon representing the occupancy of the vehicle."""
        xy, th = points_with_orient[:, :2], points_with_orient[:, 2]
        # take the average angle for the intermediate points to account for a nod in the polygon. The width is also bigger
        th_diffs = th[2:] - th[1:-1]
        th[1:-1] += th_diffs / 2
        w_scale_interm_points = 1 / np.cos(th_diffs / 2)
        left_ws = np.hstack(([left_width], w_scale_interm_points * left_width, [left_width]))
        right_ws = np.hstack(([right_width], w_scale_interm_points * right_width, [right_width]))
        # Calculate the points that make up the polygon
        perp_vecs = np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)]).T
        left_points, right_points = xy + perp_vecs * left_ws.reshape(-1, 1), xy - perp_vecs * right_ws.reshape(-1, 1)

        # The polygon might intersect with itself at to the start and end point
        d_end = np.hypot(*np.diff(xy[-2:], axis=0).T)
        d_start = np.hypot(*np.diff(xy[:2], axis=0).T)
        d_th_end = th[-1] - th[-2]
        d_th_start = th[1] - th[0]

        left_points = left_points if np.tan(d_th_end) * left_width < d_end else left_points[:-1]
        right_points = right_points if np.tan(-d_th_end) * right_width < d_end else right_points[:-1]
        left_points = left_points if np.tan(d_th_start) * left_width < d_start else left_points[1:]
        right_points = right_points if np.tan(-d_th_start) * right_width < d_start else right_points[1:]

        if not left_points.size or not right_points.size:
            raise FlatOccupancy
        return Polygon(np.vstack((right_points, np.flip(left_points, axis=0))))

    def create_occupancy_set(
        self,
        velocity_profile: Union[
            npt.NDArray[npt.Shape["N"], npt.Float],
            npt.NDArray[npt.Shape["N, 2"], npt.Float],
        ],
        loc_pos_error: float = 0,
        loc_orient_error: float = 0,
        traj_long_error: float = 0,
        traj_lat_error: float = 0,
        traj_orient_error: float = 0,
    ) -> SetBasedPrediction:
        """Custom class that generates a occupancy prediction for every timestep that accounts for uncertainty.

        Important notes:
            - First waypoint should be current position to account for deviation from the centerline
            - Check somehow (maybe in hindsight) that the occupancies stay in its own lane.

        Errors that are accounted for:
            - Localization error: position error and orientation error
            - Velocity error: Lower limit is 0, since only positive motor values are sent -> maybe make relative.
            - Trajectory following error: longitudinal, lateral and orientation error

        Arguments:
            velocity_profile -- The velocity profile with a velocity or range of velocities for each point on the
                horizon. [m/s]
            waypoints -- The waypoints along which to plan the trajectory [m]
            loc_pos_error -- The magnitude of the error on the ego vehicle position to account for [m]
            loc_orient_error -- The magnitude of the error on the ego vehicle orienation to account for [rad]
            traj_long_error -- The magnitude of the tracking error of the trajectory in the longitudinal direction [m]
            traj_lat_error -- The magnitude of the tracking error of the trajectory in the lateral direction [m]
            traj_orient_error -- The magnitude of the orientation error of the trajectory in the lateral direction [m]
            vehicle_size -- The vehicle dimension: (length, width) [m]
            vehicle_center_offset -- distance the rectangle shape has to move from the actual center point of the robot to
                fit the robot. So positive values if the actual center point is more on the back and the left side of the
                robot.

        Returns:
            An overapproximation of the possible occupancies for the given velocity profile
        """
        veh_front_l, veh_back_l = (
            self.vehicle_size[0] / 2 + self.vehicle_center_offset[0],
            self.vehicle_size[0] / 2 - self.vehicle_center_offset[0],
        )
        veh_left_w, veh_right_w = (
            self.vehicle_size[1] / 2 - self.vehicle_center_offset[1],
            self.vehicle_size[1] / 2 + self.vehicle_center_offset[1],
        )

        # overapproximate added length and width because of the rotation errors
        orient_error_l = np.sin(loc_orient_error + traj_orient_error) * max(veh_left_w, veh_right_w) / 2
        orient_error_w = np.sin(loc_orient_error + traj_orient_error) * max(veh_front_l, veh_back_l) / 2

        # Calculate the range of reachable positions along the centerline of the path given the errors and (optionally)
        #  velocity range
        velocity_profile = np.maximum(velocity_profile, 0)
        if velocity_profile.ndim == 1:  # single velocities
            # TODO: trapezoidal integration rule
            vel_integr = np.cumsum(velocity_profile * self.dt)  # the distance along the centerline
            dist_time_max = vel_integr + veh_front_l + loc_pos_error + traj_long_error + orient_error_l
            dist_time_min = vel_integr - veh_back_l - loc_pos_error - traj_long_error + orient_error_l
            width_offset_left = veh_left_w + loc_pos_error + traj_lat_error + orient_error_w
            width_offset_right = veh_right_w + loc_pos_error + traj_lat_error + orient_error_w
        elif velocity_profile.ndim == 2:  # range of velocities
            # TODO: trapezoidal integration rule
            # the distance along centerline for highest velocity
            vel_integr_max = np.cumsum(velocity_profile[:, 1] * self.dt)
            # TODO: trapezoidal integration rule
            # the distance along centerline for lowest velocity
            vel_integr_min = np.cumsum(velocity_profile[:, 0] * self.dt)
            dist_time_max = vel_integr_max + veh_front_l + loc_pos_error + traj_long_error + orient_error_l
            dist_time_min = vel_integr_min - veh_back_l - loc_pos_error - traj_long_error + orient_error_l
            width_offset_left = veh_left_w + loc_pos_error + traj_lat_error + orient_error_w
            width_offset_right = veh_right_w + loc_pos_error + traj_lat_error + orient_error_w
        else:
            raise TypeError

        # Calculate the polygons representing the occupancies for the ranges of distance along the centerline of the
        #  path which represents the occupancy at each time step
        occupancies = []
        for idx, xy_ths in enumerate(
            self.get_xy_ths_for_range(
                dist_time_min,
                dist_time_max,
                self.dist_along_points,
                self.orient_bw_points,
                self.waypoints_w_ego_pos,
            )
        ):
            polygon = self.xy_ths_to_polygon(xy_ths, left_width=width_offset_left, right_width=width_offset_right)
            occupancies.append(Occupancy(idx + 1 + self.initial_state.time_step, polygon))  # type: ignore

        return SetBasedPrediction(self.initial_state.time_step, occupancies)  # type: ignore

    def is_safe_trajectory(self, set_based_prediction: SetBasedPrediction) -> bool:
        collision_object = create_collision_object(set_based_prediction)
        return not self.collision_checker.collide(collision_object)


def visualize_velocity_profiles(
    initial_velocity: Union[float, Interval],
    velocity_profiles: np.ndarray,
    reference_speed: Optional[float] = None,
    distance_to_goal: Optional[float] = None,
    dt: Optional[float] = None,
):
    """Visualize the velocity profiles"""
    import matplotlib.pyplot as plt

    def trapezoidal_integrate(values: npt.NDArray[npt.Shape["N"], Any], dt: float):
        mid_points = (values[:-1] + values[1:]) / 2
        return np.cumsum(mid_points * dt)

    if dt == None:
        dt = 1
        print("Set the correct dt, otherwise the calculated distance is incorrect.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ts = np.arange(velocity_profiles.shape[1] + 1) * dt
    if isinstance(initial_velocity, Scalar):
        # add current velocity for visualization
        velocity_profiles = np.hstack((np.full((len(velocity_profiles), 1), initial_velocity), velocity_profiles))

        axs[0].plot(ts, velocity_profiles.T)
        axs[0].scatter(0, initial_velocity, color="k", zorder=10, label="initial velocity")
        for v_profile in velocity_profiles:
            dist = trapezoidal_integrate(np.insert(v_profile, 0, initial_velocity), dt)
            axs[1].plot((np.arange(len(v_profile)) + 1) * dt, dist)
            axs[1].fill_between(
                (np.arange(len(v_profile)) + 1) * dt,
                np.cumsum(v_profile * dt),
                np.cumsum(v_profile * dt),
                alpha=0.2,
            )
    elif isinstance(initial_velocity, Interval):
        # add current velocity for visualization
        velocity_profiles = np.concatenate(
            (
                np.full(
                    (len(velocity_profiles), 1, 2),
                    [initial_velocity.start, initial_velocity.end],
                ),
                velocity_profiles,
            ),
            axis=1,
        )

        axs[0].plot(ts, velocity_profiles.mean(axis=2).T)
        for v_profile in velocity_profiles:
            axs[0].fill_between(ts, v_profile[:, 0], v_profile[:, 1], alpha=0.2)
        axs[0].vlines(
            0,
            initial_velocity.start,
            initial_velocity.end,
            color="k",
            linewidth=3,
            zorder=10,
            label="initial velocity",
        )

        initial_velocity_avg = (initial_velocity.start + initial_velocity.end) / 2
        for v_profile in velocity_profiles:
            dist_min = trapezoidal_integrate(np.insert(v_profile[:, 0], 0, initial_velocity.start), dt)
            dist_avg = trapezoidal_integrate(np.insert(v_profile.mean(axis=1), 0, initial_velocity_avg), dt)
            dist_max = trapezoidal_integrate(np.insert(v_profile[:, 1], 0, initial_velocity.end), dt)

            axs[1].plot(ts, dist_avg)
            axs[1].fill_between(ts, dist_min, dist_max, alpha=0.2)

    axs[0].hlines(
        [0],
        0,
        len(velocity_profiles[0]) * dt,
        color="k",
        linestyles="--",
        label="velocity limits",
    )
    if reference_speed is not None:
        axs[0].hlines(
            [reference_speed],
            0,
            len(velocity_profiles[0]) * dt,
            color="k",
            linestyles="--",
        )
    if distance_to_goal is not None:
        axs[1].hlines(
            distance_to_goal,
            0,
            len(velocity_profiles[0]) * dt,
            color="k",
            linestyle="--",
            label="distance to goal",
        )

    for ax in axs:
        ax.grid()
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
    axs[0].set_title("Velocity profiles")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("velocity [m/s]")
    axs[1].set_title(f"Distance travelled (dt = {dt} s)")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("distance [m]")
    plt.tight_layout()
    plt.show()
