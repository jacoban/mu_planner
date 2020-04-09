import matplotlib.pyplot as plt

import random
import math
import numpy as np

from motion import get_intermediate_configurations_from_primitive_sequence

from footprint import Footprint

def plot_solution(problem, solution=None, explored_states=None):

    if explored_states is not None:
        for state in explored_states:
            state.footprint.plot('y')

    if solution is not None:
        for node in solution:
            node.state.footprint.plot('g')
        plt.xlabel("Path cost: " + str(round(solution[-1].path_cost, 2)))

    plt.axis('square')

    plt.xlim(0, problem.scene.x_max)
    plt.ylim(0, problem.scene.y_max)