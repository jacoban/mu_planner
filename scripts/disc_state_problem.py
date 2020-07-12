import math
import problem


class DiscState(problem.State):
    """
    Discretized state
    """

    def __init__(self, footprint, x, y, theta, prev=None):
        super().__init__(footprint, x, y, theta, prev)

        #  hybrid-A* approximation
        self.x_cell = math.ceil(x / problem.XY_DISC)
        self.y_cell = math.ceil(y / problem.XY_DISC)
        self.theta_cell = math.floor(theta % (2 * math.pi) / problem.THETA_DISC)

    def __eq__(self, other):
        assert isinstance(other, DiscState)
        return self.x_cell == other.x_cell and self.y_cell == other.y_cell and self.theta_cell == other.theta_cell

    def __lt__(self, other):
        assert isinstance(other, DiscState)

        return self.x_cell < other.x_cell or (self.x_cell == other.x_cell and self.y_cell < other.y_cell) or \
               (self.x_cell == other.x_cell and self.y_cell == other.y_cell and self.theta_cell < other.theta_cell)

    def __hash__(self):
        return hash(tuple([self.x_cell, self.y_cell, self.theta_cell]))


class Problem(problem.Problem):

    def __init__(self, initial, goal, scene, motion_primitives,
                 delta_t, velocity, vehicle_shape,
                 max_crash_prob, sampling_algorithm):
        super().__init__(initial, goal, scene, motion_primitives,
                       delta_t, velocity, vehicle_shape,
                       max_crash_prob, sampling_algorithm)

        self.discrete = True

    def result(self, state, action):
        return DiscState(action.footprint, action.x, action.y, action.theta, state)
