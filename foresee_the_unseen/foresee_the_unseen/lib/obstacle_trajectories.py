import yaml
import os
import numpy as np
from typing import Tuple, Dict, List, Optional
import nptyping as npt

from commonroad.scenario.state import InitialState
from commonroad.scenario.trajectory import Trajectory as TrajectoryCR
from commonroad.scenario.scenario import Lanelet, LaneletNetwork
from shapely.geometry import MultiPoint, Point as ShapelyPoint, Polygon as ShapelyPolygon, LineString

from rclpy.time import Duration, Time
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Point as PointMsg,
    Quaternion,
    Pose,
    Vector3,
    Twist,
    Accel,
    PoseWithCovarianceStamped,
)
from nav_msgs.msg import Path
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg


"""A lot of this code is partly copied from the Planner in foresee_the_unseen/lib/planner.py and probably other files"""


def Lanelet2ShapelyPolygon(lanelet):
    assert isinstance(lanelet, Lanelet)
    right = lanelet.right_vertices
    left = np.flip(lanelet.left_vertices, axis=0)
    lanelet_boundary = np.concatenate((right, left, np.array([right[0]])))

    lanelet_shapely = ShapelyPolygon(lanelet_boundary)
    if not lanelet_shapely.is_valid:
        lanelet_shapely = lanelet_shapely.buffer(0)
        if not lanelet_shapely.is_valid:
            print(
                "Note: Shape of lanelet",
                lanelet.lanelet_id,
                "is not valid, creating valid shape with convex hull of lane boundary.",
            )
            lanelet_shapely = MultiPoint(lanelet_boundary).convex_hull
            assert lanelet_shapely.is_valid, "Failed to convert lanelet to polygon"
    return lanelet_shapely


def generate_waypoints(
    lanelet_network: LaneletNetwork,
    initial_position: npt.NDArray[npt.Shape["2"], npt.Float],
    movement: str,
    waypoint_distance: float,
    reference_velocity: float,
    acceleration_limit: float,
    goal_positions_per_movement: Dict[str, List[List[float]]],
) -> Tuple[
    npt.NDArray[npt.Shape["N, 2"], npt.Float],
    npt.NDArray[npt.Shape["N"], npt.Float],
    npt.NDArray[npt.Shape["N"], npt.Float],
]:

    goal_positions = goal_positions_per_movement[movement]
    # the ids of the lanelets the vehicle is currently on
    for goal_position in goal_positions:
        goal_position = np.array(goal_position)
        goal_position_ids = lanelet_network.find_lanelet_by_position([goal_position])[0]
        assert len(goal_position_ids) != 0, f"Goal position not on a lane: {goal_position=}"

    # the ids of the lanelets the vehicle is currently on
    starting_lanelet_ids = lanelet_network.find_lanelet_by_position([initial_position])[0]
    assert len(starting_lanelet_ids) != 0, f"Position not on a lane {initial_position=}"
    assert len(starting_lanelet_ids) == 1, "On multiple lanelets"

    # the actual lanelets the vehicle is currently on
    starting_lanelets = []
    for lanelet_id in starting_lanelet_ids:
        starting_lanelets.append(lanelet_network.find_lanelet_by_id(lanelet_id))

    trajectory_lane = None
    for lanelet in starting_lanelets:
        # All the lanes (serie of lanelets) that can be followed by starting on the specific lanelet
        possible_lanes = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanelet_network)[0]
        # Check if any of this possible lanes intersect with any of the goal points.
        for lane in possible_lanes:
            lane_shape = Lanelet2ShapelyPolygon(lane)
            for goal_position in goal_positions:
                if lane_shape.intersects(ShapelyPoint(goal_position)):
                    trajectory_lane = lane
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        print(trajectory_lane)
        assert False, "Goal and start position not on the same lane"

    waypoints = trajectory_lane.center_vertices

    # get the intial and goal distance along the centerline
    centerline_shapely = LineString(trajectory_lane.center_vertices)
    dist_init = centerline_shapely.project(ShapelyPoint(initial_position))
    dist_goal = centerline_shapely.project(ShapelyPoint(goal_position))

    # get equally spaced waypoints
    dist_bw_points = np.insert(np.cumsum(np.hypot(*np.diff(waypoints, axis=0).T)), 0, 0)
    equally_spaced_points_x = np.interp(
        np.linspace(dist_init, dist_goal, int(dist_bw_points[-1] / waypoint_distance) + 1),
        dist_bw_points,
        waypoints[:, 0],
    )
    equally_spaced_points_y = np.interp(
        np.linspace(dist_init, dist_goal, int(dist_bw_points[-1] / waypoint_distance) + 1),
        dist_bw_points,
        waypoints[:, 1],
    )
    new_waypoints = np.vstack((equally_spaced_points_x, equally_spaced_points_y)).T

    # Get the orientation for each waypoint
    yaws = np.arctan2(*np.diff(new_waypoints, axis=0).T[::-1])
    yaws_at_points = (yaws[:-1] + yaws[1:]) / 2
    orientations = np.insert(yaws_at_points, [0, -1], [yaws[0], yaws[-1]])

    # Get the velocity for each waypoint
    velocities = np.full(len(new_waypoints), reference_velocity)
    # make a ramp at the beginning and the end:
    #  [x = a * t^2 / 2] and [v = a * t], so: [v = sqrt(2 * x * a)] and [x = v^2 / (2 * a)]
    #  [v = sqrt(2 * x * a)] can be used to find the velocity at the waypoints given some acceleration limit
    #  [x = v^2 / (2 * a)] can be used to calculate the distance when the maximum velocity is reached
    ramp_length_dist = reference_velocity**2 / (2 * acceleration_limit)
    ramp_length_idx = int(ramp_length_dist / waypoint_distance) + 1
    ramp_velocities = np.sqrt(2 * np.arange(ramp_length_idx) * waypoint_distance * acceleration_limit)
    velocities[:ramp_length_idx] = ramp_velocities
    velocities[-ramp_length_idx:] = np.flip(ramp_velocities)

    # [1:] excluded start position and zero goal velocity
    return new_waypoints[1:], orientations[1:], velocities[1:]


def generate_velocity_profile(
    waypoints: np.ndarray,
    velocity: float,
    max_acc: float,
    max_dec: float,
    dt: float,
) -> np.ndarray:
    """Make a velocity profile for a array of waypoints"""
    dist_to_goal = np.sum(np.hypot(*np.diff(waypoints, axis=0).T))
    acc_profile = np.full(int(100 / dt), -max_dec)
    v_reachable = np.sqrt(2 * (dist_to_goal + velocity**2 / (2 * max_acc)) * max_acc * max_dec / (max_acc + max_dec))

    if v_reachable < velocity:
        #                       <- reference velocity (not reached)
        # v-profile:   /\
        #                \
        t_increasing = v_reachable / max_acc
        num_incr_steps = t_increasing / dt

        acc_profile[: int(num_incr_steps)] = max_acc
        acc_profile[int(num_incr_steps)] = max_acc * (num_incr_steps % 1) + max_dec * (num_incr_steps % -1)
    else:
        #              .--.     <- reference velocity (reached)
        # v-profile:  /    \
        #                   \
        t_increasing = velocity / max_acc
        num_incr_steps = t_increasing / dt
        t_horizontal = (dist_to_goal - velocity**2 / (2 * max_acc) - velocity**2 / (2 * max_dec)) / velocity
        num_horz_steps = t_horizontal / dt
        t_decreasing = velocity / max_dec
        num_decr_steps = t_decreasing / dt

        acc_profile[: int(num_incr_steps)] = max_acc  # increasing velocity
        acc_profile[int(num_incr_steps)] = num_incr_steps % 1 * max_acc  # smaller acc to just reach reference speed
        acc_profile[int(num_incr_steps) + 1 : int(num_incr_steps + num_horz_steps)] = 0  # constant velocity
        acc_profile[-int(num_decr_steps) - 1 :] = -max_dec  # ensure that decelerating part is long enough

    v_profile = np.cumsum(acc_profile * dt)

    return np.append(v_profile[: np.argmax(v_profile < 0)], 0)


def velocity_profile_to_state_list(
    velocity_profile: np.ndarray,
    waypoints: np.ndarray,
    dt: float,
    max_dist_corner_smoothing: float = 0.1,
):
    v_diffs = velocity_profile - np.insert(velocity_profile[:-1], 0, 0)
    accelerations = v_diffs / dt

    diff_points = np.diff(waypoints, axis=0)
    dist_bw_points = np.hypot(*diff_points.T)
    dist_along_points = np.hstack(([0], np.cumsum(dist_bw_points)))
    orient_bw_points = np.arctan2(*diff_points.T[::-1])

    # use the 'trapezoidal rule' for integrating the velocity
    # TODO: trapezoidal integration rule
    # velocity_extend = np.insert(velocity_profile, 0, self.initial_state.velocity)
    # velocity_trapez = (velocity_extend[:-1] + velocity_extend[1:]) / 2
    # dist = np.cumsum(velocity_trapez * self.dt)
    dist = np.cumsum(velocity_profile * dt)
    x_coords = np.interp(dist, dist_along_points, waypoints[:, 0])
    y_coords = np.interp(dist, dist_along_points, waypoints[:, 1])

    # The trajectory orientation on the parts between the waypoints is constant, but the trajectory  orientation
    #  around the waypoints is smoothed. The smoothening happens within a distance of the minimum of
    #  (`self.max_dist_corner_smoothing`) and half of the length of the part between two waypoints.
    dist_along_bef_points = dist_along_points[1:-1] - np.minimum(
        np.diff(dist_along_points[:-1]) / 2, max_dist_corner_smoothing
    )
    dist_along_aft_points = dist_along_points[1:-1] + np.minimum(
        np.diff(dist_along_points[1:]) / 2, max_dist_corner_smoothing
    )
    dist_along_bef_aft_points = np.vstack((dist_along_bef_points, dist_along_aft_points)).T.flatten()
    orientations_at_points = np.repeat(orient_bw_points, 2)[1:-1]
    orientations = np.interp(dist, dist_along_bef_aft_points, orientations_at_points)

    state_list = []
    for time_step, (x_coord, y_coord, orientation, velocity, acceleration) in enumerate(
        zip(x_coords, y_coords, orientations, velocity_profile, accelerations, strict=True)
    ):
        state = InitialState(
            position=np.array([x_coord, y_coord]),
            orientation=orientation,
            velocity=velocity,
            acceleration=acceleration,
            time_step=time_step + 1,  # type: ignore
        )
        state_list.append(state)

    return TrajectoryCR(1, state_list)  # type: ignore


def quaternion_from_yaw(yaw):
    return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]


def get_ros_trajectory_from_commonroad_trajectory(
    trajectory: TrajectoryCR, start_stamp: Time, dt: float
) -> TrajectoryMsg:
    """Publishes the Commonroad trajectory on a topic for the trajectory follower node."""
    header_path = Header(stamp=start_stamp.to_msg(), frame_id="planner")
    pose_stamped_list = []
    init_time_step = trajectory.state_list[0].time_step - 1
    for state in trajectory.state_list:
        quaternion = Quaternion(**{k: v for k, v in zip(["x", "y", "z", "w"], quaternion_from_yaw(state.orientation))})
        position = PointMsg(x=float(state.position[0]), y=float(state.position[1]))
        time_diff = (state.time_step - init_time_step) * dt
        header_pose = Header(
            stamp=(start_stamp + Duration(seconds=int(time_diff), nanoseconds=int((time_diff % 1) * 1e9))).to_msg()
        )
        pose_stamped_list.append(PoseStamped(header=header_pose, pose=Pose(position=position, orientation=quaternion)))
    twist_list = [Twist(linear=Vector3(x=float(s.velocity))) for s in trajectory.state_list]
    if trajectory.state_list[0].acceleration is not None:
        accel_list = [Accel(linear=Vector3(x=float(s.acceleration))) for s in trajectory.state_list]
    else:
        accel_list = []
    trajectory_msg = TrajectoryMsg(
        path=Path(header=header_path, poses=pose_stamped_list), velocities=twist_list, accelerations=accel_list
    )

    return trajectory_msg


def to_ros_trajectory_msg(
    waypoints: npt.NDArray[npt.Shape["N, 2"], npt.Float],
    orientations: npt.NDArray[npt.Shape["N"], npt.Float],
    velocities: npt.NDArray[npt.Shape["N"], npt.Float],
) -> TrajectoryMsg:
    """Returns a ROS Trajectory message which only contains the waypoints and velocities."""
    header_path = Header(frame_id="planner")
    pose_stamped_list = []
    for waypoint, orientation in zip(waypoints, orientations, strict=True):
        pose = PoseStamped(
            pose=Pose(
                position=PointMsg(x=waypoint[0], y=waypoint[1]),
                orientation=Quaternion(z=np.sin(orientation / 2), w=np.cos(orientation / 2)),
            )
        )
        pose_stamped_list.append(pose)

    twist_list = [Twist(linear=Vector3(x=float(v))) for v in velocities]

    return TrajectoryMsg(path=Path(header=header_path, poses=pose_stamped_list), velocities=twist_list)


def get_ros_pose_from_commonroad_state(state: InitialState) -> PoseWithCovarianceStamped:
    """Commonroad InitialState to ROS message PoseWithCovarianceStamped"""
    x, y = state.position[0], state.position[1]
    yaw = state.orientation
    msg = PoseWithCovarianceStamped()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = np.sin(yaw / 2)
    msg.pose.pose.orientation.w = np.cos(yaw / 2)

    return msg

def read_obstacle_configuration_yaml(yaml_file: str, namespace: Optional[str] = None) -> Dict:
    with open(yaml_file) as f:
        if namespace:
            obstacle_config = yaml.safe_load(f)[namespace]
        else:
            obstacle_config = yaml.safe_load(f)
    assert os.path.isfile(
        obstacle_config["road_structure_xml"]
    ), f"`road_structure_xml` should be a file: {type(obstacle_config['road_structure_xml'])=}"

    goal_positions_per_movement = {k: np.array(v) for k, v in obstacle_config["movements"].items()}
    assert np.all(
        [v.ndim == 2 and v.shape[1] == 2 for v in goal_positions_per_movement.values()]
    ), "shape of the goal positions for each movement should be [N, 2]"
    waypoint_distance = obstacle_config["waypoint_distance"]
    assert isinstance(
        waypoint_distance, (float, int)
    ), f"`waypoint_distance` should be a scalar number {type(waypoint_distance)=}"

    for value in obstacle_config["obstacle_cars"].values():
        assert np.array(value["start_pose"]).shape == (
            3,
        ), f"The start position should be a 3D array: {value['start_pose']=}"
        assert (
            value["movement"] in goal_positions_per_movement.keys()
        ), f"Movement ({value['movement']=}) should be one of {goal_positions_per_movement.keys()}"
        assert isinstance(
            value["velocity"], (float, int)
        ), f"`velocity` should be a scalar number: {type(value['velocity'])=}"
        assert isinstance(
            value["acceleration"], (float, int)
        ), f"`acceleration` should be a scalar number {type(value['acceleration'])=}"
        assert isinstance(
            value["start_delay"], (float, int)
        ), f"`start_delay` should be a scalar number {type(value['start_delay'])=}"

    return obstacle_config


if __name__ == "__main__":
    waypoints = np.arange(10).reshape(-1, 2)
    dist_to_goal = np.sum(np.hypot(*np.diff(waypoints, axis=0).T))
    velocity_profile = generate_velocity_profile(
        dist_to_goal=dist_to_goal,
        velocity=0.5,
        max_acc=0.2,
        max_dec=0.2,
        dt=0.25,
    )
