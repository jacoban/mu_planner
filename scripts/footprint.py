from shapely.affinity import translate, rotate

import matplotlib.pyplot as plt


class Footprint:
    """
    Represents obstacles and vehicles
    """

    def __init__(self, shape, x, y, theta):
        shape = rotate(shape, theta, origin='centroid', use_radians=True)
        shape = translate(shape, x, y)
        self.shape = shape

    def collides(self, other):
        return self.shape.intersects(other.shape)

    def plot(self, color):
        xs, ys = self.shape.exterior.xy
        plt.plot(xs, ys, color, linewidth=.5)
