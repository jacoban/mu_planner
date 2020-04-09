import pyro.distributions as dist
import torch
from shapely.geometry import Polygon
from world import gen_world


class Scene:
    """
    Global, deterministic settings
    """

    def __init__(self, x_max, y_max, world_param, nsamples, subworld_colors):
        self.x_max = x_max
        self.y_max = y_max
        self.scene_polygon = Polygon([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]])

        self.nsamples = nsamples
        self.world_param = world_param
        self.worlds = [gen_world(self.world_param) for _ in range(self.nsamples)]
        self.world_sampler = dist.Categorical(torch.ones(len(self.worlds)))

        self.subworld_colors = subworld_colors
        assert len(world_param) == len(subworld_colors)

    def worlds_iter(self):
        while True:
            i = self.world_sampler.sample()
            yield self.worlds[i]

    def contains(self, footprint):
        return self.scene_polygon.contains(footprint.shape)

    def plot(self):
        for world in self.worlds:
            world.plot(self.subworld_colors)
