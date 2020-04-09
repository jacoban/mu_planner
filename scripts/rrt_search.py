import math
import random

from aima.search import Node


def l2_distance_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2) ** 2


def theta_distance_sq(theta1, theta2):
    return (1.0 - math.cos(theta1 - theta2)) ** 2


def get_closest_node(nodes_list, target_x, target_y, target_theta, wp=0.5, wt=0.5):
    """
    Implementation of the distance function used by Lavalle and Kuffner.
    """
    l2_dists = list(map(lambda n: l2_distance_sq(n.state.x, n.state.y, target_x, target_y), nodes_list))
    theta_dists = list(map(lambda n: theta_distance_sq(n.state.theta, target_theta), nodes_list))

    max_l2_dist = max(l2_dists)
    max_theta_dist = max(theta_dists)

    if max_theta_dist < 1e-5 or max_l2_dist < 1e-5:
        return nodes_list[0]

    wpn = wp / max_l2_dist
    wtn = wt / max_theta_dist

    dists = list(map(lambda i: wpn * l2_dists[i] + wtn * theta_dists[i], range(len(nodes_list))))

    index_min_dist = min(range(len(nodes_list)), key=lambda i: dists[i])

    return nodes_list[index_min_dist]


def rrt_search(problem, max_iter=float('inf'), goal_bias=0.2, n_saved_explored_states=0):
    """A simple rrt algorithm implemented with the use of motion primitives and using the search
    framework of the AIMA book"""

    # initial node
    n_new = Node(problem.initial)

    # list containing all the rrt nodes computed so far
    rrt = [n_new]

    it = 0

    while it < max_iter:

        if problem.goal_test(n_new.state):
            return n_new, it, list(map(lambda n: n.state, rrt))[0:n_saved_explored_states]

        if random.random() <= goal_bias:
            x_sample = problem.goal.x_center
            y_sample = problem.goal.y_center
            theta_sample = math.pi/2.0

        else:
            x_sample = random.random() * problem.scene.x_max
            y_sample = random.random() * problem.scene.y_max
            theta_sample = random.random() * 2.0 * math.pi

        n_near = get_closest_node(rrt, x_sample, y_sample, theta_sample, wp=0.5, wt=0.5)

        children = n_near.expand(problem)

        if len(children) > 0:
            n_new = get_closest_node(children, x_sample, y_sample, theta_sample, wp=0.5, wt=0.5)
            rrt.append(n_new)

        it += 1

        if it % 100 == 0:
            print("Explored nodes: " + str(it))

    return None, it, list(map(lambda n: n.state, rrt))[0:n_saved_explored_states]
