import math
import problem

XY_DISC = 0.25
THETA_DISC = math.pi / 2


class State(problem.State):
    """
    Discretized state
    """

    def __init__(self, footprint, x, y, theta, prev=None):
        super().__init__(footprint, x, y, theta, prev)

        #  hybrid-A* approximation
        self.x_cell = math.ceil(x / XY_DISC)
        self.y_cell = math.ceil(y / XY_DISC)
        self.theta_cell = math.floor(theta % (2 * math.pi) / THETA_DISC)

    def __eq__(self, other):
        assert isinstance(other, State)
        return self.x_cell == other.x_cell and self.y_cell == other.y_cell and self.theta_cell == other.theta_cell

    def __lt__(self, other):
        assert isinstance(other, State)

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

    def result(self, state, action):
        return State(action.footprint, action.x, action.y, action.theta, state)
