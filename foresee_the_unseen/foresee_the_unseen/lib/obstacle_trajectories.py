import yaml
import os
import numpy as np
from typing import Tuple, Dict, List
import nptyping as npt

from commonroad.scenario.scenario import Lanelet, LaneletNetwork
from shapely.geometry import MultiPoint, Point as ShapelyPoint, Polygon as ShapelyPolygon, LineString

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point as PointMsg, Quaternion, Pose, Vector3, Twist
from nav_msgs.msg import Path
from racing_bot_interfaces.msg import Trajectory as TrajectoryMsg


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


def read_obstacle_configuration_yaml(yaml_file: str) -> Dict:
    with open(yaml_file) as f:
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
