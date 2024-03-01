import numpy as np
import math
import copy
from typing import Union, Tuple, Any
from nptyping import NDArray, Float, Shape, assert_isinstance

from sensor_msgs.msg import LaserScan
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, MultiPolygon as ShapelyMultiPolygon


def normalize_angles(angles: Union[float, NDArray[Any, Float]]) -> Union[float, NDArray[Any, Float]]:
    """Normalizes the angles to the range [-pi, pi]

    Arguments:
        angles -- one or more angles to normalize

    Returns:
        one or more normalized angles
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def angles_ranges_to_points(angles: NDArray[Shape["N"], Float], ranges: NDArray[Shape["N"], Float]):
    """Converts angles and ranges measured by an Lidar to a cloud

    Arguments:
        angles -- the angle belonging to each range
        ranges -- the distance measured by the lidar at each angle

    Returns:
        a pointcloud
    """
    assert_isinstance(angles, NDArray[Shape["N"], Float])
    assert_isinstance(ranges, NDArray[Shape["N"], Float])
    cos_sin_map = np.array([np.cos(angles), np.sin(angles)])
    points = (ranges * cos_sin_map).T
    return points


def make_valid(
    angles: NDArray[Shape["N, 2"], Float], ranges: NDArray[Shape["N"], Float]
) -> Tuple[NDArray[Shape["N, 2"], Float], NDArray[Shape["N"], Float]]:
    """Iteratively removes or modifies sections from the FOV that are (partly) behind other sections when watched from
     the center of the FOV (the lidar).

    Arguments:
        angles -- the start and end angle for each section
        ranges -- the range for each section

    Returns:
        the angles and ranges without overlapping sections
    """
    assert_isinstance(angles, NDArray[Shape["N, 2"], Float])
    assert_isinstance(ranges, NDArray[Shape["N"], Float])
    has_start_end_overlap = angles[:, 0] > angles[:, 1]  # start angle > end angle
    if has_start_end_overlap.sum() == 0:
        return angles, ranges
    angles_valid, ranges_valid = angles[~has_start_end_overlap], ranges[~has_start_end_overlap]

    # start angle < previous end angle; account for the 2 pi difference between first and last point
    has_overlap_previous = angles_valid[:, 0] < np.hstack((angles_valid[-1:, 1] - np.pi * 2, angles_valid[:-1, 1]))
    has_smallest_range = ranges_valid[has_overlap_previous] < ranges_valid[np.roll(has_overlap_previous, -1)]
    # if has smallest range and overlap: update the end angle of the previous section
    # else has smallest range and NOT overlap: update own start angle
    # angles_valid[np.roll(has_overlp_and_small, -1), 1] = angles_valid[has_overlp_and_small, 0]
    # angles_valid[np.roll(~has_overlp_and_small), 0] = angles_valid[np.roll(~has_overlp_and_small, -1), 1]
    has_overlap_and_small = copy.deepcopy(has_overlap_previous)
    has_overlap_and_small[has_overlap_previous == True] = has_smallest_range
    has_overlap_and_not_small = copy.deepcopy(has_overlap_previous)
    has_overlap_and_not_small[has_overlap_previous == True] = ~has_smallest_range

    angles_valid[np.roll(has_overlap_and_small, -1), 1] = angles_valid[has_overlap_and_small, 0]
    angles_valid[has_overlap_and_not_small, 0] = angles_valid[np.roll(has_overlap_and_not_small, -1), 1]

    return make_valid(angles_valid, ranges_valid)


def get_underapproximation_fov(
    scan_msg: LaserScan,
    max_range=3,
    min_resolution=0.25,
    angle_margin: float = 0.0,
    use_abs_angle_margin: bool = False,
    range_margin: float = 0.0,
    use_abs_range_margin: bool = False,
    padding: float = 0.0,
) -> ShapelyPolygon:
    """Calculates the field of view (FOV) based on the measurements from a Lidar. It underapproximates the FOV and
    other margins can be applied to account for uncertainty.

    Arguments:
        scan_msg -- ROS messages returned by the Lidar sensor

    Keyword Arguments:
        max_range -- The maximum view range of the Lidar and therefore of the FOV (default: {3})
        min_resolution -- Minimum resolution in meters at the boundaries of the FOV (default: {0.25})
        angle_margin -- The margin on the rays around detected obstacles to account for uncertainty; unit is
            [m] or [rad] depending on use_abs_angle_margin (default: {0.0})
        use_abs_angle_margin -- Determines whether the angle_margin is an absolute distance to the closest object [m] or 
            a fixed angle [rad] (default: {False})
        range_margin -- The margin on the ranges to account for the uncertainty; unit is [m] or [-] (rate) depending on 
            use_abs_range_margin (default: {0.0})
        use_abs_range_margin -- Determines whether the range_margin is an absolute distance [m] or a relative 
            distance [-] (default: {False})
        padding -- Padding on the FOV [m] applied after the ray margin. To account for detection delays. (default: {0.0})

    Raises:
        AssertionError: if range_margin > 1 when using an relative margin (use_abs_range_margin == False)
        AssertionError: if multiple polygons are created after applying the padding but none of them contains the origin

    Returns:
        the polygon describing the field of view (FOV)
    """
    # """Need to account for range value uncertainty"""
    ranges = np.array(scan_msg.ranges)
    N = len(ranges)
    angles = scan_msg.angle_min + np.arange(N) * scan_msg.angle_increment
    cos_sin_map = np.array([np.cos(angles), np.sin(angles)]).T

    angle_range = scan_msg.angle_max - scan_msg.angle_min
    min_num_of_sec = math.ceil(angle_range * max_range * 2 / min_resolution)  # circumference / minimum resolution
    pts_per_sec = int(N / min_num_of_sec)
    num_of_sec = math.ceil(N / pts_per_sec)

    ranges = np.hstack((ranges, np.full(num_of_sec * pts_per_sec - N, np.inf))).reshape(num_of_sec, -1)
    ranges_min = ranges.min(axis=1)
    ranges_min[ranges_min > max_range] = max_range

    if use_abs_range_margin:
        ranges_min = np.clip(ranges_min - range_margin, 0, None)
    else:
        assert range_margin < 1, f"The margin on the ranges should be smaller than 1: range_margin = {range_margin}"
        ranges_min *= (1 - range_margin)

    diff_ranges_backward = ranges_min - np.roll(ranges_min, 1)  # compare to previous range
    mask_extend_backward = diff_ranges_backward < 0
    diff_ranges_forward = ranges_min - np.roll(ranges_min, -1)  # compare to next range
    mask_extend_forward = diff_ranges_forward < 0

    angles = np.hstack((angles, np.full(num_of_sec * pts_per_sec - N, angles[-1]))).reshape(num_of_sec, -1)
    angle_range_per_sec = angles[:, [0, -1]]

    # angle margin dependant on range; so ray should always be x [m] from closest point
    dist_margin = angle_margin
    if use_abs_angle_margin:
        angle_margin_backward = np.arcsin(dist_margin / 2 / ranges_min[mask_extend_backward]) * 2
        angle_margin_forward = np.arcsin(dist_margin / 2 / ranges_min[mask_extend_forward]) * 2
    else:
        angle_margin_backward = angle_margin
        angle_margin_forward = angle_margin

    # add 1 angle_increment and the margin to the smallest ranges
    angle_range_per_sec[mask_extend_backward, 0] -= scan_msg.angle_increment + angle_margin_backward
    angle_range_per_sec[mask_extend_forward, 1] += scan_msg.angle_increment + angle_margin_forward

    # substract the margin from the longest ranges
    angle_range_per_sec[np.roll(mask_extend_backward, -1), 1] -= angle_margin_backward
    angle_range_per_sec[np.roll(mask_extend_forward, 1), 0] += angle_margin_forward

    # remove invisible parts due to the angle margin
    angles, ranges = make_valid(angle_range_per_sec, ranges_min)

    points = angles_ranges_to_points(angles.flatten(), np.repeat(ranges, 2))
    shapely_polygon = ShapelyPolygon(points)
    if abs(padding) > 1e-10: # do not apply near 0 paddings
        shapely_polygon = shapely_polygon.buffer(-padding)

    # polygon might split into multiple polygons when applying a buffer
    if isinstance(shapely_polygon, ShapelyMultiPolygon):
        origin = ShapelyPoint(0, 0)
        for polygon in list(shapely_polygon):
            if polygon.contains(origin):
                return polygon
        else:
            assert False, "None of the polygons contains the origin"
    else:
        return shapely_polygon
