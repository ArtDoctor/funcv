import numpy as np
from numpy.typing import NDArray


def create_square(size: float) -> NDArray[np.float64]:
    s = size / 2
    return np.array([
        [-s, -s, 0],
        [s, -s, 0],
        [s, s, 0],
        [-s, s, 0]
    ], dtype=np.float64)


def rotate_y(points: NDArray[np.float64], ay: float) -> NDArray[np.float64]:
    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ], dtype=np.float64)
    return points @ Ry.T


def project(points3D: NDArray[np.float64], movement: NDArray[np.float64]) -> NDArray[np.float64]:
    projected = []
    for pt in points3D + movement:
        x, y, z = pt
        f = 300
        scale = f / (f + z + 500)
        x2d = int(x * scale + 320)
        y2d = int(y * scale + 240)
        projected.append([x2d, y2d, z])
    return np.array(projected, dtype=np.float64) 