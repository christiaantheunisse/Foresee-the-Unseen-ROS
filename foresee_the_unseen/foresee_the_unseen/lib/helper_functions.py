import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


def polygons_from_road_xml(xml_file: str, offset: np.ndarray = np.array([0., 0.])):
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


# if __name__ =="__main__":
#     polygons = polygons_from_road_xml("road_structure.xml")

#     for lanelet_id, polygon in polygons.items():
#         plt.plot(*polygon.T, label=lanelet_id)
#     plt.legend()
#     plt.show()
