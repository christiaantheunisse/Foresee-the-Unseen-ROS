import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import transforms
from dataclasses import dataclass

from typing import List, Union


@dataclass
class Trajectory:
    xys: np.ndarray  # xy-positions [m] [[x, y], ...]
    yaws: np.ndarray  # headings [rad] [yaw, ...]
    vs: np.ndarray  # velocties [m/s] [v, ...]

    def __add__(self, other):
        return Trajectory(
            xys=np.vstack((self.xys, other.xys)),
            yaws=np.hstack((self.yaws, other.yaws)),
            vs=np.hstack((self.vs, other.vs)),
        )

    def __getitem__(self, items):
        if isinstance(items, float):
            assert abs(items) <= 1.0, "float should be between -1 and 1"
            if items >= 0:
                items = slice(0, round(items * len(self)))
            else:
                items = slice(round(items * len(self)), len(self))
        elif isinstance(items, tuple):
            if len(items) == 2 and isinstance(items[0], float) and isinstance(items[1], float):
                items = slice(round(items[0] * len(self)), round(items[1] * len(self)))
        return Trajectory(
            xys=self.xys[items],
            yaws=self.yaws[items],
            vs=self.vs[items],
        )

    def __len__(self):
        return len(self.xys)

    def translate(self, translation: Union[np.ndarray, List] = [0, 0]):
        new = copy.deepcopy(self)
        new.xys = new.xys + translation

        return new

    def rotate(self, theta: float = 0):
        new = copy.deepcopy(self)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        new.xys = (rot_matrix @ new.xys.T).T
        new.yaws = new.yaws + theta

        return new

    def mirror_x(self):
        new = copy.deepcopy(self)
        new.xys = new.xys * [1, -1]  # mirror y-coordinates
        new.yaws = -new.yaws  # mirror orientation around x-axis

        return new

    def mirror_y(self):
        new = copy.deepcopy(self)
        new.xys = new.xys * [-1, 1]  # mirror x-coordinates
        new.yaws = -(new.yaws + np.pi / 2) - np.pi / 2  # mirror orientation around y-axis

        return new

    def repeat(self, repetitions: int = 10):
        new = copy.deepcopy(self)
        new.xys = np.tile(new.xys, (repetitions, 1))
        new.yaws = np.tile(new.yaws, repetitions)
        new.vs = np.repeat(new.vs, repetitions)

        return new

    def visualize(self):
        # first of all, the base transformation of the data points is needed
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)

        plt.plot((0, 1), (0, 0), color=(1, 0, 0), transform=rot + base)
        plt.plot((0, 0), (0, 1), color=(0, 1, 0), transform=rot + base)
        plt.scatter(*self.xys.T, c=np.arange(len(self.xys)), cmap="cool", label="states", transform=rot + base)

        # orientation vector
        orientation_vectors = np.array([np.cos(self.yaws), np.sin(self.yaws)]).T * self.vs.reshape(-1, 1)
        filter_o_vectors = 5
        for xy, o in zip(self.xys[::filter_o_vectors], orientation_vectors[::filter_o_vectors]):
            plt.plot(*np.array([xy, xy + o]).T, "k--", transform=rot + base)
            # plt.scatter(*xy, color='k', transform= rot + base)

        plt.grid()
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()


def get_straight_line_trajectory(velocity: float = 0.4):
    no_points = 100
    xys = np.linspace([0, 0], [5, 0], no_points)
    diffs = xys[1:] - xys[:-1]
    diffs = np.append(diffs, diffs[-1:], axis=0)
    yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
    vs = np.full(no_points, velocity)

    return Trajectory(xys, yaws, vs)


def get_circular_trajectory(repetitions: int = 10, radius: float = 0.8, velocity: float = 0.4):
    no_points = 100
    thetas = np.linspace(np.pi / 2, -np.pi * 3 / 2, no_points + 1)[:-1]
    xys = np.array([np.cos(thetas), np.sin(thetas)]).T * radius + np.array([0, -radius])
    yaws = thetas - np.pi / 2
    vs = np.full(no_points, velocity)

    return Trajectory(xys, yaws, vs).repeat(repetitions)


def get_circular_reverse_trajectory(repetitions: int = 10, radius: float = 0.8, velocity: float = 0.4):
    trajectory = get_circular_trajectory(repetitions=repetitions, radius=radius, velocity=velocity)

    return trajectory.mirror_y().mirror_x()


def get_eight_trajectory(repetitions: int = 10, radius=0.8, velocity: float = 0.4):
    circ_trajectory = get_circular_trajectory(repetitions=1, radius=radius, velocity=velocity)
    return (circ_trajectory + circ_trajectory.mirror_x()).repeat(repetitions=repetitions)


def get_sinus_trajectory(x_shape: float = 2.0, y_shape: float = 2.0, velocity: float = 0.4):
    no_points = 100
    thetas = np.linspace(0, np.pi * 2, no_points)
    xs = np.linspace(0, x_shape, no_points)
    ys = np.sin(thetas) * y_shape / 2
    xys = np.vstack((xs, ys)).T
    yaws = yaws_from_xys(xys)
    vs = np.full(no_points, velocity)

    return Trajectory(xys, yaws, vs)


def get_sinus_and_circ_trajectory(radius=1, velocity=0.4):
    rot_sin = get_sinus_trajectory(x_shape=2 * radius, y_shape=2 * radius, velocity=velocity).rotate(-np.pi / 2)
    circ = get_circular_trajectory(radius=radius, velocity=velocity, repetitions=1)

    return circ[(0., 0.25)] + \
        rot_sin[(0.25, 0.75)].translate([0, -radius/2]) + \
            circ[(0.25, 0.75)].mirror_y().translate([0, -radius]) + \
                rot_sin[(0.25, 0.75)].translate([0, radius]).mirror_x().translate([0, -radius * 3/2]) + \
                circ[(0.75, 1.)]



def yaws_from_xys(xys):
    diffs = xys[1:] - xys[:-1]
    diffs = np.append(diffs, diffs[-1:], axis=0)
    return np.arctan2(diffs[:, 1], diffs[:, 0])


if __name__ == "__main__":
    # get_straight_line_trajectory().visualize()
    radius = 1
    velocity = 1
    # get_sinus_trajectory(x_shape=2 * radius, y_shape=2 * radius, velocity=velocity).rotate(-np.pi / 2).translate([0, radius]).mirror_x().translate([0, -radius * 3/2]).visualize()
    get_sinus_and_circ_trajectory().visualize()
    # get_circular_trajectory(repetitions=1)[(0.25, 0.75)].translate([0, 0.8]).rotate(np.pi / 2).visualize()
    # get_sinus_trajectory().visualize()
    # get_circular_reverse_trajectory(repetitions=1).visualize()
    # get_eight_trajectory(repetitions=1).visualize()
    # get_eight_trajectory(repetitions=5).visualize()
