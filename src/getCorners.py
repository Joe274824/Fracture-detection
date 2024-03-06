

import numpy as np


def getCorners(center, orientation_matrix, size, imgMins):
    corner_points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corner = center + (i * orientation_matrix[:, 0] * (size[0] / 2)) + (j * orientation_matrix[:, 1] * (size[1] / 2)) + (k * orientation_matrix[:, 2] * (size[2] / 2))
                corner_points.append(corner - imgMins)

    return corner_points

if __name__ == "__main__":
    center = np.array([0.5, 0.5, 0.5])
    orientation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    size = np.array([1, 1, 1])
    imgMins = np.array([0, 0, 0])
    print(getCorners(center, orientation_matrix, size, imgMins))

    print(getCorners([50, 50, 50], np.identity(3), [100, 100, 100], [0, 0, 0]))