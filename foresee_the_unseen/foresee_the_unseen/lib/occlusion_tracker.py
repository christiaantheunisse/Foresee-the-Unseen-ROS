
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LinearRing
from shapely.ops import substring

from commonroad.scenario.scenario import Scenario, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import InitialState
from commonroad.prediction.prediction import SetBasedPrediction, Occupancy

from utilities import Lanelet2ShapelyPolygon, ShapelyPolygon2Polygon, polygon_intersection, polygon_diff, polygon_union, cut_line

import time

class Shadow:
    def __init__(self,
                 polygon,
                 lane):
        self.polygon = polygon
        self.lane = lane
        self.center_line = ShapelyLineString(self.lane.center_vertices)
        self.right_line = ShapelyLineString(self.lane.right_vertices)
        self.left_line = ShapelyLineString(self.lane.left_vertices)
        self.lane_shapely = Lanelet2ShapelyPolygon(lane)

    def expand(self, dist):
        self.polygon = self.__get_next_occ(self.polygon, dist)

    def get_occupancy_set(self, time_step, dt, max_vel, prediction_horizon, planning_horizon = 100):
        dist = dt*max_vel
        occupancy_set = []
        #pred_polygon_shapely = self.polygon

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
        red_factor = 5
        assert prediction_horizon % red_factor == 0, \
            f"Number of steps on the prediction horizon should be dividable by {red_factor}"
        red_prediction_horizon = prediction_horizon // red_factor
        dist *= red_factor

        for i in range(prediction_horizon):
            # Extend the top edges without overpasing the length of the lane
            #   the front and rear of the prediction sets are always made perpendicular to the path / flat.
            if i % 5 == 0:
                new_top_right = max(top_right + dist, self.right_line.project(self.left_line.interpolate(top_left + dist)))
                new_top_left = max(top_left + dist, self.left_line.project(self.right_line.interpolate(top_right + dist)))
                top_right = new_top_right
                top_left = new_top_left
                top_right = min(top_right, self.right_line.length)
                top_left = min(top_left, self.left_line.length)
                
                # print(f"top_right: {top_right}")
                # print(f"top_left: {top_left}")

                pred_polygon_shapely = self.__build_polygon(bottom_right, bottom_left, top_right, top_left)
                pred_polygon = ShapelyPolygon2Polygon(pred_polygon_shapely)
            
            occupancy = Occupancy(time_step+i+1, pred_polygon)
            # occupancy_set.extend([occupancy for _ in range(red_factor)])
            occupancy_set.append(occupancy)
        
        # Populate the rest of the planning horizon with the last prediction
        for i in range(prediction_horizon, planning_horizon):
            occupancy = Occupancy(time_step+i+1, pred_polygon)
            occupancy_set.append(occupancy)
        
        return occupancy_set

    def __get_next_occ(self, poly, dist):
        smallest_projection = 999999
        for edge in poly.exterior.coords:
            projection = self.center_line.project(ShapelyPoint(edge))
            if projection < smallest_projection:
                smallest_projection = projection
            if smallest_projection <= 0:
                break
        poly = poly.buffer(dist, join_style=1)
        intersection = polygon_intersection(poly, self.lane_shapely)
        poly = intersection[0] #This has to be fixed

        if smallest_projection > 0:
            sub_center_line = substring(self.center_line, 0, smallest_projection)
            left_side = sub_center_line.parallel_offset(2.8, 'left')
            right_side = sub_center_line.parallel_offset(2.8, 'right')
            Area_to_substract = ShapelyPolygon(np.concatenate((np.array(left_side.coords), np.array(right_side.coords))))
            diff = polygon_diff(poly, Area_to_substract)
            poly = diff[0] #This has to be fixed

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
        if not isinstance(shadow_shapely, ShapelyPolygon):#, "shadow_boundary: " + str(shadow_boundary)
            print(type(shadow_shapely))
            print("Not instance")
            assert LinearRing(shadow_boundary).is_valid
        return shadow_shapely

class Occlusion_tracker:
    def __init__(self,
                 scenario,
                 initial_time_step=0,
                 min_vel=0,
                 max_vel=1,
                 #min_acc=-1,
                 #max_acc=1,
                 min_shadow_area = 1,
                 initial_sensor_view = ShapelyPolygon(),
                 prediction_horizon = 10,
                 tracking_enabled = True):
        self.time_step = initial_time_step
        self.dt = scenario.dt
        self.min_vel = min_vel
        self.max_vel = max_vel
        #self.min_acc = min_acc
        #self.max_acc = max_acc
        self.min_shadow_area = min_shadow_area
        self.shadows = []
        self.prediction_horizon = prediction_horizon
        self.tracking_enabled = tracking_enabled

        ## Find the initial lanelets
        initial_lanelets = []
        for lanelet in scenario.lanelet_network.lanelets:
            if lanelet.predecessor == []:
                initial_lanelets.append(lanelet)

        ## Generate lanes (Collection of lanelets from start to end of the scenario)
        lanes = []
        for lanelet in initial_lanelets:
            current_lanes, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, scenario.lanelet_network, max_length=500)
            for lane in current_lanes:
                lanes.append(lane)
        self.lanes = lanes

        # ========== Find only the 3 relevant lanes for the `parked_vehicle_scenario` ==========
        lanelets_dict = {}
        for lanelet in scenario.lanelet_network.lanelets:
            lanelets_dict[lanelet.lanelet_id] = lanelet

        self.lanes = []
        if scenario.scenario_id.map_name == "MyIntersection":
            _lane = Lanelet.merge_lanelets(lanelets_dict[17], lanelets_dict[16])
            self.lanes.append(Lanelet.merge_lanelets(_lane, lanelets_dict[15]))
            self.lanes.append(Lanelet.merge_lanelets(lanelets_dict[17], lanelets_dict[30]))
            self.lanes.append(Lanelet.merge_lanelets(lanelets_dict[12], lanelets_dict[13]))
        elif scenario.scenario_id.map_name == "Ffb":
            _lane = Lanelet.merge_lanelets(lanelets_dict[49574], lanelets_dict[49600])
            self.lanes.append(Lanelet.merge_lanelets(_lane, lanelets_dict[49566]))
            self.lanes.append(Lanelet.merge_lanelets(lanelets_dict[49574], lanelets_dict[49590]))
            self.lanes.append(Lanelet.merge_lanelets(lanelets_dict[49564], lanelets_dict[49602]))

        # my scenario

        # ==================== END ====================

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

        #plt.figure()
        #plot(scenario, shadows=self.shadows)
        #plt.show()

    def update(self, sensor_view, new_time):
        if self.tracking_enabled == True:
            self.update_tracker(sensor_view, new_time)
        else:
            self.reset(sensor_view, new_time)

    def update_tracker(self, sensor_view, new_time):
        assert(new_time>=self.time_step)
        time_diff = new_time - self.time_step
        # Update the time
        self.time_step = new_time
        # Expand all the shadows
        for shadow in self.shadows:
            shadow.expand(self.dt*self.max_vel*time_diff)

        # Intersect them with the current sensorview
        new_shadows = []
        for shadow in self.shadows:
            intersections = polygon_diff(shadow.polygon, sensor_view)
            if not intersections:
                pass
            else:
                for intersection in intersections:
                    assert intersection.is_valid
                    assert not intersection.is_empty
                    if intersection.area >= self.min_shadow_area:
                        new_shadows.append(Shadow(intersection, shadow.lane))
        self.shadows = new_shadows

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def reset(self, sensor_view=ShapelyPolygon(), new_time=0):
        # Update the time
        self.time_step = new_time
        # Reset all the shadows
        new_shadows = []
        for lane in self.lanes:
            lanelet_shapely = Lanelet2ShapelyPolygon(lane)
            shadow_polygons = polygon_diff(lanelet_shapely, sensor_view)
            for shadow_polygon in shadow_polygons:
                if shadow_polygon.area >= self.min_shadow_area:
                    #print(shadow_polygon.area)
                    #plt.figure()
                    #plot(shapelyPolygons=[shadow_polygon])
                    #plt.show()
                    current_shadow = Shadow(shadow_polygon, lane)
                    new_shadows.append(current_shadow)
        self.shadows = new_shadows

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def get_dynamic_obstacles(self, scenario):
        dynamic_obstacles = []

        # print(f"There are {len(self.shadows)} shadows")
        # print(f"Timestep: {self.time_step}, dt: {self.dt}")
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # # cmap = cm.get_cmap('hot')
        # # color = cmap((self.time_step / 3) % 1)
        # fig, axs = plt.subplots(int(len(self.shadows) / 3) + 1, 3)
        # for idx, shadow in enumerate(self.shadows):
        #     # plt.plot(*shadow.polygon.exterior.xy, color=color)
        #     ax = axs.flatten()[idx]
        #     ax.plot(*shadow.polygon.exterior.xy)
        #     ax.set_xlim(60, 200)
        #     ax.set_ylim(-13, 13)
        #     ax.set_aspect('equal')
        #     # ax.set_xlim(0, 100)
        #     # ax.set_ylim(-40, 40)
        # plt.show()

        import time
        time_steps = []
        for shadow in self.shadows:
            t_steps = [time.time()]  # LOG RUNTIME
            occupancies = []
            occupancy_set = shadow.get_occupancy_set(self.time_step, self.dt, self.max_vel, self.prediction_horizon)
            t_steps.append(time.time())  # LOG RUNTIME
            obstacle_id = scenario.generate_object_id()
            obstacle_type = ObstacleType.UNKNOWN
            t_steps.append(time.time())  # LOG RUNTIME
            obstacle_shape = ShapelyPolygon2Polygon(shadow.polygon)
            obstacle_initial_state = InitialState(position = np.array([0,0]),
                                           velocity = self.max_vel,
                                           orientation = 0,
                                           time_step = self.time_step)
            t_steps.append(time.time())  # LOG RUNTIME
            obstacle_prediction = SetBasedPrediction(self.time_step+1, occupancy_set)
            t_steps.append(time.time())  # LOG RUNTIME
            dynamic_obstacle = DynamicObstacle(obstacle_id,
                            obstacle_type,
                            obstacle_shape,
                            obstacle_initial_state,
                            obstacle_prediction)
            dynamic_obstacles.append(dynamic_obstacle)
            t_steps.append(time.time())  # LOG RUNTIME
            t_steps = np.array(t_steps)
            t_steps = t_steps[1:] - t_steps[:-1]
            time_steps.append(t_steps)

        time_steps = np.array(time_steps)
        # print(f"[shadows] Time per step:\n{time_steps.mean(axis=0).tolist()}")
        # print(f"[shadows] Percentage of time per step:\n{(time_steps.mean(axis=0) / (time_steps.sum() / time_steps.shape[0]) * 100).tolist()}")      
        # print("")

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
