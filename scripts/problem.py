"""
This file contains the definition of the planning problem
"""

import math

import shapely.speedups

import aima
from aima.utils import distance

from motion import get_new_configurations_from_primitives

from footprint import Footprint

import sprt
import ztest


MINSAMPLES = dict([(0.01, 211), (0.1, 103), (0.25, 100)])
MAXSAMPLES = dict([(0.01, 225), (0.1, 225), (0.25, 225)])


class State:
    """
    The robot's configuration (i.e., 2D coordinates and orientation).
    """

    def __init__(self, footprint, x, y, theta, prev = None):
        self.footprint = footprint
        self.x = x
        self.y = y
        self.theta = theta % (math.pi * 2)

        # the immediately preceding state in the trajectory
        self.prev = prev

    def __eq__(self, other):
        assert isinstance(other, State)
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __hash__(self):
        return hash(tuple([self.x, self.y, self.theta]))


class Action:

    def __init__(self, footprint, x, y, theta):
        self.footprint = footprint
        self.x = x
        self.y = y
        self.theta = theta

class Goal:

    def __init__(self, x_min, x_max, y_min, y_max, theta_min, theta_max):
        assert x_min < x_max
        assert y_min < y_max
        assert 0 <= theta_min < theta_max <= math.pi * 2

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.theta_min = theta_min
        self.theta_max = theta_max

        self.x_center = (x_min + x_max) / 2.0
        self.y_center = (y_min + y_max) / 2.0


def world_collides_state(world, state):
    """
    Return true iff the trajectory represented by 'state' does not collide with 'world'
    """

    curr_state = state

    while curr_state is not None:
        if world.collides(curr_state.footprint):
            return True

        curr_state = curr_state.prev

    return False

def world_collides_new_state(world, state, new_footprint):
    return world_collides_state(world, state) or world.collides(new_footprint)


class Problem(aima.search.Problem):
    """
    Subclasses the abstract Problem class in aima/search.py.
    """

    def __init__(self, initial, goal, scene, motion_primitives,
                 delta_t, velocity, vehicle_shape, 
                 max_crash_prob, hyptest):

        super().__init__(initial, goal)

        # the global, deterministic settings
        self.scene = scene

        # pre-computed motion primitives
        self.motion_primitives = motion_primitives

        # 2D span of the robot w.r.t. the plane origin at the beginning of the search
        self.vehicle_shape = vehicle_shape

        # time needed to execute an action
        self.delta_t = delta_t

        # fixed linear speed of the motion primitives
        self.velocity = velocity

        # the maximal collision probability tolerated
        self.max_crash_prob = max_crash_prob

        # the hypothesis test
        self.hyptest = hyptest

        if self.hyptest == 'mc':
            try:
                self.min_samples_mc = MINSAMPLES[self.max_crash_prob]
            except KeyError:
                print("Warning: using default number of samples (100) for mc sampling algorithm")
                self.min_samples_mc = 100

        try:
            self.max_samples = MAXSAMPLES[self.max_crash_prob]
        except KeyError:
            print("Warning: using default number of max samples (250)")
            self.max_samples = 250

        shapely.speedups.enable()

    def actions(self, state):
        """
        Returns candidate actions that can be applied to a given state.
        
        It is guaranteed that the robot, after applying any of the candidate actions, satisfy the following conditions:
        (1) robot remains in the map
        (2) probability of robot colliding with probabilistic
            obstacles is within the threshold.
        """

        actions = []

        configs = get_new_configurations_from_primitives(
            state.x, state.y, state.theta, self.motion_primitives)

        for mp_index in range(len(configs)):
            config = configs[mp_index]
            x = config[0]
            y = config[1]
            theta = config[2]

            new_footprint = Footprint(self.vehicle_shape, x, y, theta)

            # 1st check: the robot remains in the map

            if not self.scene.contains(new_footprint):
                continue

            # 2nd check: probability of robot crashing is within threshold

            witer = self.scene.worlds_iter()

            # boolean function to test
            def sample_no_collision():
                return not world_collides_new_state(next(witer), state, new_footprint)

            if self.hyptest == 'sprt':
                no_collision = sprt.pr_gt(sample_no_collision, prob=1 - self.max_crash_prob, alpha=0.05, beta=0.6,
                                          nsamples_max=self.max_samples)

            elif self.hyptest == 'mc':
                no_collision = ztest.pr_gt(sample_no_collision, prob=1 - self.max_crash_prob,
                                           nsamples_init=self.min_samples_mc, alpha=0.05, nsamples_max=self.max_samples)

            if no_collision:
                actions.append(Action(new_footprint, x, y, theta))

        return actions

    def result(self, state, action):
        return State(action.footprint, action.x, action.y, action.theta, state)

    def goal_test(self, state):
        goal = self.goal
        return goal.x_min <= state.x < goal.x_max and \
               goal.y_min <= state.y < goal.y_max and \
               goal.theta_min <= state.theta < goal.theta_max

    def path_cost(self, c, state1, action, state2):
        """
        Traveled distance
        """
        return c + self.velocity * self.delta_t

    def value(self, state):
        raise NotImplementedError

    def h(self, node):
        """
        Euclidean distance heuristic
        """
        return distance((node.state.x, node.state.y), (self.goal.x_center, self.goal.y_center))
