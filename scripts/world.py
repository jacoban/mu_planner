import pyro
import pyro.distributions as dist
from torch import tensor

from footprint import Footprint

class World:
    """
    A possbile world consists of a set of subworlds that are superimposed to form a world.
    """

    def __init__(self, subworlds):
        self.subworlds = subworlds
        self.collisions_cache = {}
        self.num_nominal_coll_checks = 0
        self.num_actual_coll_checks = 0

    def collides(self, footprint):
        self.num_nominal_coll_checks += 1

        # This computation is memoized 
        cached_result = self.collisions_cache.get(footprint)
        if cached_result is None:
            self.num_actual_coll_checks += 1

            collides = False
            for subworld in self.subworlds:
                if subworld.collides(footprint):
                    collides = True
            self.collisions_cache[footprint] = collides
            return collides
        else:
            return cached_result

        # for subworld in self.subworlds:
        #     if subworld.collides(footprint):
        #         return True
        # return False

    def plot(self, colors):
        assert len(self.subworlds) == len(colors)

        for subworld, scenario_colors in zip(self.subworlds, colors):
            subworld.plot(scenario_colors)


class SubWorld:
    """
    A subworld consists of a set of obstacles.
    """

    def __init__(self, obstacles, scenario_index):
        self.obstacles = obstacles
        self.scenario_index = scenario_index

    def collides(self, footprint):
        for obs in self.obstacles:
            if obs.collides(footprint):
                return True
        return False

    def plot(self, colors):
        for obs in self.obstacles:
            obs.plot(colors[self.scenario_index])


def gen_obstacle(g_mean, g_cov, shape):
    """
    Returns an Obstacle instance by sampling according to the obstacle's generative model.

    Args:
        g_mean (length-3 list): the mean of the obstacle's 2D coordinates and orientation
        g_cov  (length-3 list of length-3 lists): the covariance of the obstacle's 2D coordinates and orientation
    """

    d = dist.MultivariateNormal(tensor(g_mean), tensor(g_cov))
    # sample the obstacle's geometrics
    x, y, theta = d.sample().tolist()
    return Footprint(shape, x, y, theta)


def gen_subworld(likelihoods, scenarios):
    """
    Returns a SubWorld instance by sampling according to the subworld's generative model.

    Args:
        likelihoods (list of floats): the likelihoods of the disjoint scenarios of the subworld
        scenarios (list of lists of 3-tuples)
    """

    assert len(likelihoods) == len(scenarios)

    # sample a scenario
    d = dist.Categorical(tensor(likelihoods))
    i = d.sample()
    obstacles = []
    # sample all obstacles in the scenario
    for g_mean, g_cov, shape in scenarios[i]:
        obstacles.append(gen_obstacle(g_mean, g_cov, shape))
    return SubWorld(obstacles, i)


def gen_world(subworld_params):
    """
    Returns a World instance by sampling according to the world's generative model

    Args:
        subworld_params (list of pairs)
    """

    subworlds = []
    for likelihoods, scenarios in subworld_params:
        subworlds.append(gen_subworld(likelihoods, scenarios))
    return World(subworlds)
