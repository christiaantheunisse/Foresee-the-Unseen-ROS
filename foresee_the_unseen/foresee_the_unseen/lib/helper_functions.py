import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
from collections import namedtuple


def polygons_from_road_xml(xml_file: str, offset: np.ndarray = np.array([0.0, 0.0])):
    tree = ET.parse(xml_file)

    root = tree.getroot()
    lanelets = root.findall("lanelet")
    polygons_dict = {}
    for lanelet in lanelets:
        lanelet_id = lanelet.get("id")
        # print(f"Lanelet id: {lanelet_id}")
        left_bound = lanelet.find("leftBound").findall("point")
        right_bound = lanelet.find("rightBound").findall("point")

        left_points, right_points = [], []
        for point in left_bound:
            left_points.append([point.find("x").text, point.find("y").text])
        for point in right_bound:
            right_points.append([point.find("x").text, point.find("y").text])
        left_points, right_points = np.array(left_points), np.array(right_points)

        # polygon = np.vstack((left_points, np.flip(right_points, axis=0), left_points[0:1])).astype(np.float_)
        polygon = np.vstack((left_points, np.flip(right_points, axis=0))).astype(np.float_) + offset
        if lanelet_id not in polygons_dict:
            polygons_dict[lanelet_id] = polygon
        else:
            print("None unique lanelet_id")

    return polygons_dict


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def matrix_from_transform(t):
    rot_quat = [
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
    ]
    translation = [
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z,
    ]
    rot_scipy = R.from_quat(rot_quat)
    rot_matrix = rot_scipy.as_matrix()
    t_matrix = np.zeros((4, 4))
    t_matrix[:3, :3] = rot_matrix
    t_matrix[:3, 3] = translation
    t_matrix[3, 3] = 1

    return t_matrix


Point = namedtuple("Point", "x y")


def halfspace_from_points(p1: np.ndarray, p2: np.ndarray):
    assert p1.shape == p2.shape == (2,), "Points should be 2 dimensional numpy arrays"
    a = Point(*p1)
    b = Point(*p2)

    # RHS of the line: x_coef * X + y_coef * Y <= b
    if a.x - b.x == 0:
        x_coef = a.y - b.y
        y_coef = 0
        const = -(a.x * b.y - b.x * a.y)
    else:
        x_coef = -(a.y - b.y) / (a.x - b.x)
        y_coef = 1
        const = (a.x * b.y - b.x * a.y) / (a.x - b.x)
        if a.x > b.x:
            x_coef, y_coef, const = -x_coef, -y_coef, -const

    return x_coef, y_coef, const

def matrices_from_cw_cvx_polygon(polygon):
    polygon = np.array(polygon)
    A, B = [], []
    for p1, p2 in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b, c = halfspace_from_points(p1, p2)
        A.append([a, b])
        B.append(c)

    return np.array(A), np.array(B)


# if __name__ =="__main__":
#     polygons = polygons_from_road_xml("road_structure.xml")

#     for lanelet_id, polygon in polygons.items():
#         plt.plot(*polygon.T, label=lanelet_id)
#     plt.legend()
#     plt.show()
