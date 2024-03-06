import numpy as np
import math
from typing import Optional
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import InitialState
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)

from foresee_the_unseen.lib.utilities import Lanelet2ShapelyPolygon
from shapely.geometry import Point
from shapely.geometry import LineString


class PositionNotOnALane(Exception):
    """The position is not on a lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class GoalAndPositionNotSameLane(Exception):
    """The goal and the position are not on the same lane."""

    def __init__(self, message: Optional[str] = None):
        self.message = message


class Planner:
    def __init__(
        self,
        initial_state,
        goal_point,
        vehicle_shape,
        reference_speed,
        max_acceleration,
        max_deceleration,
        time_horizon,
        dt,
        min_dist_waypoint,
        # goal_point=[],
        # vehicle_shape=Rectangle(4.8, 2),
        # reference_speed=9,
        # max_acceleration=2,
        # max_deceleration=4,
        # time_horizon=50,
        # dt=0.1,
        # min_dist_waypoint=0.1,
        waypoints=[],
    ):
        self.initial_state = initial_state
        self.waypoints = list(waypoints)
        self.goal_point = goal_point
        self.vehicle_shape = vehicle_shape
        self.reference_speed = reference_speed
        self.max_acc = max_acceleration
        self.max_dec = np.abs(max_deceleration)
        self.time_horizon = time_horizon
        self.dt = dt
        self.min_dist_waypoint = min_dist_waypoint

    def update(self, state):
        self.initial_state = state
        if self.waypoints:
            self.remove_passed_waypoints()

    def plan(self, scenario):
        trajectories = self.generate_trajectories(scenario.lanelet_network)
        safe_trajectories = self.get_safe_trajectories(trajectories, scenario)
        optimal_trajectory = self.get_optimal_trajectory(safe_trajectories)
        return optimal_trajectory

    def generate_trajectories(self, lanelet_network):
        if not self.waypoints:
            self.find_waypoints(lanelet_network)

        velocity_profiles = self.generate_velocity_profiles()
        trajectories = self.create_trajectories(velocity_profiles)
        return trajectories

    def find_waypoints(self, lanelet_network):
        starting_lanelet_ids = lanelet_network.find_lanelet_by_position([self.initial_state.position])[0]
        if starting_lanelet_ids == []:
            raise PositionNotOnALane(f"Current position: {self.initial_state.position}")

        starting_lanelets = []
        for lanelet_id in starting_lanelet_ids:
            starting_lanelets.append(lanelet_network.find_lanelet_by_id(lanelet_id))

        starting_lane = None
        for lanelet in starting_lanelets:
            starting_lanes = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanelet_network)[0]
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

        # remove waypoints beyond goal position.
        center_vertices = np.array(starting_lane.center_vertices)
        center_line_shapely = LineString(center_vertices)
        goal_shapely = Point(self.goal_point)
        dist_along_line = center_line_shapely.project(goal_shapely)
        dists = np.cumsum(np.linalg.norm(center_vertices[1:] - center_vertices[:-1], axis=1))
        mask = dists < dist_along_line
        mask = np.insert(mask, 0, True)

        self.all_possible_waypoints = list(starting_lane.center_vertices)
        self.waypoints = np.array(center_vertices[mask])
        self.waypoints = np.vstack((self.waypoints, np.array(self.goal_point).reshape(1, -1)))

        # Add extra waypoints between points that are to far apart
        # TODO: ugly!!!!, maybe change planner so that it also works between points
        max_dist = 0.2
        dists = np.linalg.norm(self.waypoints[1:] - self.waypoints[:-1], axis=1)
        points_to_add = (dists / max_dist).astype(np.int_)
        dist_add = dists / (points_to_add + 1)
        new_waypoints = []
        for w1, w2, num, dist in zip(self.waypoints[:-1], self.waypoints[1:], points_to_add, dist_add):
            new_waypoints.append(w1)
            diff_vec = (w2 - w1) / np.linalg.norm(w2 - w1) * dist
            new_waypoints.extend([w1 + (i + 1) * diff_vec for i in range(num)])

        new_waypoints.append(self.waypoints[-1])
        self.waypoints = new_waypoints

        self.remove_passed_waypoints()

    def remove_passed_waypoints(self):
        while True:
            if not self.waypoints:
                self.waypoints = [self.goal_point]
                break
            direction_vector = self.waypoints[0] - self.initial_state.position
            angle_to_next_point = np.arctan2(*np.flip(direction_vector))
            angle_diff_to_next_point = abs(
                (np.pi + angle_to_next_point - self.initial_state.orientation) % (2 * np.pi) - np.pi
            )
            next_point_is_too_close = bool(np.hypot(*direction_vector) < self.min_dist_waypoint)
            if next_point_is_too_close or (angle_diff_to_next_point > np.pi / 2):
                self.waypoints.pop(0)
            else:
                break

        self.waypoints.insert(0, self.initial_state.position)

    def generate_velocity_profiles(self, number_of_trajectories=10):

        # Accelerate immediately or decelerate after one time step
        velocity_decs = self.initial_state.velocity - self.max_dec * self.dt * np.arange(self.time_horizon)
        velocity_incs = self.initial_state.velocity + self.max_acc * self.dt * (1 + np.arange(self.time_horizon))
        max_velocity_increase = abs(velocity_decs[-1])
        assert velocity_decs[-1] <= 0, (
            "Planning horizon too short for current speed to reach 0!\n",
            f"current velocity = {self.initial_state.velocity}\n",
            f"max deceleratie = {self.max_dec}",
            f"max dt = {self.dt}",
            f"max time_horizon = {self.time_horizon}",
            f"velocity_decs = {velocity_decs}",
        )

        # FIXME: cannot handle current velocity > reference speed -> temporary fix -> should reshape trajectory
        reference_speed = max(self.reference_speed, self.initial_state.velocity)
        velocity_profiles = []
        for velocity_increase in np.linspace(max_velocity_increase, 0, number_of_trajectories):
            unbounded_velocity_profile = np.minimum(velocity_incs, velocity_increase + velocity_decs)
            velocity_profile = np.clip(unbounded_velocity_profile, 0, reference_speed)
            velocity_profiles.append(velocity_profile)
        assert np.array_equal(velocity_profiles[-1], velocity_decs.clip(0))
        return velocity_profiles

    def create_trajectories(self, velocity_profiles):
        trajectories = []
        for velocities in velocity_profiles:
            trajectories.append(self.create_trajectory(velocities))
        return trajectories

    def create_trajectory(self, velocities):
        start_time_step = self.initial_state.time_step + 1

        distance_along_time = np.cumsum(velocities * self.dt)

        direction_vectors_between_points = np.diff(self.waypoints, axis=0)
        x_diffs, y_diffs = zip(*direction_vectors_between_points)

        distance_along_points = np.concatenate(([0], np.cumsum(np.hypot(y_diffs, x_diffs))))

        # This gives the x and y coordinate for each step of the velocity profile
        x_points, y_points = zip(*self.waypoints)
        x_along_time = np.interp(distance_along_time, distance_along_points, x_points)
        y_along_time = np.interp(distance_along_time, distance_along_points, y_points)

        orientations_between_points = np.unwrap(np.arctan2(y_diffs, x_diffs))
        # Some room for improvement: each point has the orientation of the previous hypot
        orientations_along_time = np.interp(distance_along_time, distance_along_points[1:], orientations_between_points)
        state_list = []
        for time_step, velocity in enumerate(velocities):
            position = np.array([x_along_time[time_step], y_along_time[time_step]])
            orientation = orientations_along_time[time_step]
            state = InitialState(
                position=position,
                orientation=orientation,
                velocity=float(velocity),
                time_step=time_step + start_time_step,
            )
            state_list.append(state)
        return Trajectory(start_time_step, state_list)

    def get_safe_trajectories(self, trajectories, scenario):
        safe_trajectories = []
        collision_checker = create_collision_checker(scenario)
        for trajectory in trajectories:
            prediction = TrajectoryPrediction(trajectory, self.vehicle_shape)
            collision_obj = create_collision_object(prediction)
            if not collision_checker.collide(collision_obj):
                safe_trajectories.append(prediction)
        return safe_trajectories

    def get_optimal_trajectory(self, trajectories):
        optimal_trajectory = []
        # Currently ordered so best trajectory is first
        if trajectories:
            optimal_trajectory = trajectories[0]
        return optimal_trajectory
