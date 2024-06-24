import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import copy

from shapely import BufferJoinStyle
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    LineString as ShapelyLineString,
    Point as ShapelyPoint,
    LinearRing,
)
from shapely.ops import substring
from shapely.errors import GEOSException

from commonroad.scenario.scenario import Scenario, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState
from commonroad.prediction.prediction import SetBasedPrediction, Occupancy

from foresee_the_unseen.lib.utilities import (
    Lanelet2ShapelyPolygon,
    ShapelyPolygon2Polygon,
    polygon_intersection,
    polygon_diff,
    polygon_union,
    cut_line,
)


def recursive_merge(lanelets: list[Lanelet]) -> Lanelet:
    assert len(lanelets) > 0
    if len(lanelets) == 1:
        return lanelets[0]
    elif len(lanelets) == 2:
        return Lanelet.merge_lanelets(*lanelets)
    else:
        lane = Lanelet.merge_lanelets(lanelets[0], lanelets[1])
        for lanelet in lanelets[2:]:
            lane = lanelet.merge_lanelets(lane, lanelet)
        return lane


class Shadow:
    def __init__(self, polygon: ShapelyPolygon, lane: Lanelet):
        self.polygon = polygon
        self.lane = lane
        self.center_line = ShapelyLineString(self.lane.center_vertices)
        self.right_line = ShapelyLineString(self.lane.right_vertices)
        self.left_line = ShapelyLineString(self.lane.left_vertices)
        self.lane_shapely = Lanelet2ShapelyPolygon(lane)

    def expand(self, dist: float):
        if abs(dist) > 1e-10:
            new_polygon = self.__get_next_occ(self.polygon, dist)
            self.polygon = new_polygon if new_polygon is not None else self.polygon

    def get_occupancy_set(
        self, time_step: int, dt: float, max_vel: float, prediction_horizon: int, steps_per_occ_pred: int
    ):
        dist = dt * max_vel
        occupancy_set = []
        # pred_polygon_shapely = self.polygon

        # Calculate the right and left projections
        right_projections = []
        left_projections = []

        for edge in self.polygon.exterior.coords:
            right_projections.append(self.right_line.project(ShapelyPoint(edge)))
            left_projections.append(self.left_line.project(ShapelyPoint(edge)))

        # Calculate the edges of the current shadow
        bottom_right = min(right_projections)
        bottom_left = min(left_projections)
        top_right = max(right_projections)
        top_left = max(left_projections)

        # print(f"bottom_right: {bottom_right}")
        # print(f"top_right: {top_right}")
        # print(f"bottom_left: {bottom_left}")
        # print(f"top_left: {top_left}")
        # print('-'*50)

        ### THIS IS THE MOST TIME CONSUMING STEP IN THE WHOLE CODE
        # reduce the number of predictions steps, but keep the same horizon
        assert prediction_horizon % steps_per_occ_pred == 0, (
            f"Prediction horizon should be dividable by the steps to combine in the occupancy prediction: "
            + f"prediction_horizon = {prediction_horizon}, steps_per_occ_pred = {steps_per_occ_pred}"
        )
        dist *= steps_per_occ_pred

        for pred_step in range(int(prediction_horizon / steps_per_occ_pred)):
            # Extend the top edges without overpasing the length of the lane
            #   the front and rear of the prediction sets are always made perpendicular to the path / flat.
            new_top_right = max(top_right + dist, self.right_line.project(self.left_line.interpolate(top_left + dist)))
            new_top_left = max(top_left + dist, self.left_line.project(self.right_line.interpolate(top_right + dist)))
            top_right = new_top_right
            top_left = new_top_left
            top_right = min(top_right, self.right_line.length)
            top_left = min(top_left, self.left_line.length)

            pred_polygon_shapely = self.__build_polygon(bottom_right, bottom_left, top_right, top_left)
            pred_polygon = ShapelyPolygon2Polygon(pred_polygon_shapely)

            occupancy_set.extend(
                [
                    Occupancy(time_step + pred_step * steps_per_occ_pred + i, pred_polygon)
                    for i in range(steps_per_occ_pred)
                ]
            )

        return occupancy_set

    def __get_next_occ(self, poly: ShapelyPolygon, dist: float):
        smallest_projection = 999999
        for edge in poly.exterior.coords:
            projection = self.center_line.project(
                ShapelyPoint(edge)
            )  # project the edges of the occupancy polygon on the centerline
            if projection < smallest_projection:
                smallest_projection = projection
            if smallest_projection <= 0:
                break
        poly = poly.buffer(dist, join_style=BufferJoinStyle.mitre)
        intersection = polygon_intersection(poly, self.lane_shapely)
        poly = intersection[0]  # This has to be fixed

        if smallest_projection > 0:  # if the starting point of the occlusion is not the starting point of the lane
            sub_center_line = substring(self.center_line, 0, smallest_projection)
            left_side = sub_center_line.parallel_offset(2.8, "left")  # FIXME: hardcoded maximum lane width
            right_side = sub_center_line.parallel_offset(2.8, "right")  # FIXME: hardcoded maximum lane width
            Area_to_substract = ShapelyPolygon(
                # The orientation of the left_side and right_side seems to be similar, so one of them needs to be reversed.
                #  At least in some situations, this causes and error so for now, I'll reverse one of them
                np.concatenate((np.array(left_side.coords), np.flip(np.array(right_side.coords), axis=0)))
            )
            diff = polygon_diff(poly, Area_to_substract)
            poly = diff[0] if diff else None  # This has to be fixed

        return poly

    def __build_polygon(self, bottom_right, bottom_left, top_right, top_left):
        # Cut the left and right lines with the top and bottom points
        right_side = cut_line(self.right_line, bottom_right, top_right)
        left_side = cut_line(self.left_line, bottom_left, top_left)

        # Build the polygon
        left_side.reverse()
        shadow_boundary = right_side + left_side + [right_side[0]]
        shadow_shapely = ShapelyPolygon(shadow_boundary)

        shadow_shapely = shadow_shapely.buffer(0)

        assert shadow_shapely.is_valid
        assert not shadow_shapely.is_empty
        if not isinstance(shadow_shapely, ShapelyPolygon):  # , "shadow_boundary: " + str(shadow_boundary)
            print(type(shadow_shapely))
            print("Not instance")
            assert LinearRing(shadow_boundary).is_valid
        return shadow_shapely


class Occlusion_tracker:
    def __init__(
        self,
        scenario: Scenario,
        min_vel: float,
        max_vel: float,
        # min_acc=-1,
        # max_acc=1,
        min_shadow_area: float,
        prediction_horizon: int,
        steps_per_occ_pred: int,  # no. of time steps to combine in the occupancy prediction
        dt: float,
        initial_sensor_view: ShapelyPolygon = ShapelyPolygon(),
        initial_time_step: int = 0,
        tracking_enabled: bool = True,
        lanes_to_merge: Optional[list[list[int]]] = None,
    ):
        self.time_step = initial_time_step
        self.dt = dt
        self.min_vel = min_vel
        self.max_vel = max_vel
        # self.min_acc = min_acc
        # self.max_acc = max_acc
        self.min_shadow_area = min_shadow_area
        self.shadows: list[Shadow] = []
        self.prediction_horizon = prediction_horizon
        self.tracking_enabled = tracking_enabled
        self.steps_per_occ_pred = steps_per_occ_pred

        lanelets_dict = {}
        for lanelet in scenario.lanelet_network.lanelets:
            lanelets_dict[lanelet.lanelet_id] = lanelet

        self.lanes = []
        if lanes_to_merge is not None:
            self.lanes = [recursive_merge([lanelets_dict[l] for l in lanelets]) for lanelets in lanes_to_merge]
        else:
            ## Find the initial lanelets
            initial_lanelets = []
            for lanelet in scenario.lanelet_network.lanelets:
                if lanelet.predecessor == []:
                    initial_lanelets.append(lanelet)

            ## Generate lanes (Collection of lanelets from start to end of the scenario)
            lanes = []
            for lanelet in initial_lanelets:
                current_lanes, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                    lanelet, scenario.lanelet_network, max_length=500
                )
                for lane in current_lanes:
                    lanes.append(lane)
            self.lanes = lanes

        # Calculate the first "view"
        for lane in self.lanes:
            lanelet_shapely = Lanelet2ShapelyPolygon(lane)
            shadow_polygons = polygon_diff(lanelet_shapely, initial_sensor_view)
            for shadow_polygon in shadow_polygons:
                if shadow_polygon.area >= self.min_shadow_area:
                    current_shadow = Shadow(shadow_polygon, lane)
                    self.shadows.append(current_shadow)

        # Calculate the first occluded area
        self.accumulated_occluded_area = 0

    def update(self, sensor_view: ShapelyPolygon, new_time_step: int, scan_delay: float) -> None:
        """
        new_time: is the planner step
        scan_delay: is time in seconds ago the scan was made/started.
        """
        if self.tracking_enabled == True:
            self.update_tracker(sensor_view, new_time_step, scan_delay)
        else:
            self.reset(sensor_view, new_time_step, scan_delay)

    def update_tracker(self, sensor_view: ShapelyPolygon, new_time_step: int, scan_delay: float) -> None:
        """
        new_time_step: is the planner step
        scan_delay: is time in seconds ago the scan was made/started.
        """
        assert new_time_step >= self.time_step
        time_diff = (new_time_step - self.time_step) * self.dt
        # Update the time
        self.time_step = new_time_step

        # This option is over conservative, because: if the scan delay is bigger than the time difference with the
        #  previous step, the shadows increasing more than the time passed since the previous step. This is necessary
        #  since the scan is older.
        # time_before_scan = max(time_diff - scan_delay, 0)
        # time_after_scan = max(scan_delay, 0)

        # for shadow in self.shadows:
        #     shadow.expand(self.max_vel * time_before_scan)

        # # Expand all the shadows
        for shadow in self.shadows:
            shadow.expand(time_diff * self.max_vel)
        # # Negatively buffer the FOV to account for delay
        sensor_view_formal = sensor_view.buffer(-scan_delay * self.max_vel)
        # sensor_view_formal = sensor_view

        # Intersect them with the current sensorview
        new_shadows = []
        for shadow in self.shadows:
            intersections = polygon_diff(shadow.polygon, sensor_view_formal)
            if not intersections:
                # new_shadows.append(shadow)
                continue
            else:
                for intersection in intersections:
                    assert intersection.is_valid
                    assert not intersection.is_empty
                    if intersection.area >= self.min_shadow_area:
                        new_shadows.append(Shadow(intersection, shadow.lane))
        self.shadows = self.prune_shadows(new_shadows)

        # extend the shadows to account for the scan delay
        # if time_after_scan > 0.:
        #     for shadow in self.shadows:
        #         shadow.expand(self.max_vel * time_after_scan)

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def prune_shadows(self, shadows: list[Shadow]) -> list[Shadow]:
        """Reduce the total number of obstacles by merging obstacles that overlap."""
        shadows = np.array(shadows)
        merged_shadows = []
        lane_ids = np.array([shadow.lane.lanelet_id for shadow in shadows])
        for lane_id in np.unique(lane_ids):
            shadows_to_union = shadows[lane_ids == lane_id]
            lane = shadows_to_union[0].lane
            merged_polygons = polygon_union([shadow.polygon for shadow in shadows_to_union])
            merged_shadows.extend([Shadow(polygon, lane) for polygon in merged_polygons])
        return merged_shadows

    def reset(self, sensor_view: ShapelyPolygon, new_time_step: int, scan_delay: float):
        # Update the time
        self.time_step = new_time_step

        sensor_view_formal = sensor_view.buffer(-scan_delay * self.max_vel)

        # Reset all the shadows
        new_shadows = []
        for lane in self.lanes:
            lanelet_shapely = Lanelet2ShapelyPolygon(lane)
            shadow_polygons = polygon_diff(lanelet_shapely, sensor_view_formal)
            for shadow_polygon in shadow_polygons:
                if shadow_polygon.area >= self.min_shadow_area:
                    # print(shadow_polygon.area)
                    # plt.figure()
                    # plot(shapelyPolygons=[shadow_polygon])
                    # plt.show()
                    current_shadow = Shadow(shadow_polygon, lane)
                    new_shadows.append(current_shadow)
        self.shadows = new_shadows

        # extend the shadows to account for the scan delay
        # if scan_delay > 0.:
        #     for shadow in self.shadows:
        #         shadow.expand(self.max_vel * scan_delay)

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def get_dynamic_obstacles(self, scenario) -> list[DynamicObstacle]:
        dynamic_obstacles = []

        for shadow in self.shadows:
            occupancy_set = shadow.get_occupancy_set(
                self.time_step, self.dt, self.max_vel, self.prediction_horizon, self.steps_per_occ_pred
            )
            obstacle_id = scenario.generate_object_id()
            obstacle_type = ObstacleType.UNKNOWN
            obstacle_shape = ShapelyPolygon2Polygon(shadow.polygon)
            obstacle_initial_state = InitialState(
                position=np.array([0, 0]), velocity=self.max_vel, orientation=0, time_step=self.time_step
            )
            obstacle_prediction = SetBasedPrediction(self.time_step + 1, occupancy_set)
            dynamic_obstacle = DynamicObstacle(
                obstacle_id, obstacle_type, obstacle_shape, obstacle_initial_state, obstacle_prediction
            )
            dynamic_obstacles.append(dynamic_obstacle)

        return dynamic_obstacles

    def get_currently_occluded_area(self):
        # Get all the shadow polygons:
        list_of_shadows = []
        for shadow in self.shadows:
            list_of_shadows.append(shadow.polygon)

        # Calculate the union:
        polygon_list = polygon_union(list_of_shadows)

        # Add up all the areas:
        currently_occluded_area = 0
        for polygon in polygon_list:
            currently_occluded_area = currently_occluded_area + polygon.area

        # Return the currently occluded area:
        return currently_occluded_area
