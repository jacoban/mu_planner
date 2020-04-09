import math
import time

from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from aima.search import astar_search
from rrt_search import rrt_search

from footprint import Footprint
from scene import Scene
from motion import get_motion_primitives_diff_drive

from problem import Goal
import problem
import disc_state_problem

from plot import plot_solution

import argparse

import pyro


parser = argparse.ArgumentParser(description='Planner')
parser.add_argument("-a", "--algorithm", type=int, choices=[
                    0, 1], default=0, help="search algorithm to use; 0 (default) is A*, 1 is RRT")
parser.add_argument("--nodisc", action="store_true",
                    help="no discretization of search space")
parser.add_argument("-t", "--test", type=str, default='sprt', choices=["sprt", "ztest"], action="store", help="SPRT or Z-Test")
parser.add_argument("-s", "--seed", type=int, default=0, help="RNG seed")
parser.add_argument("-n", "--nsamples", type=int, default=1000, help="total number of world samples")
parser.add_argument("--noplot", action="store_true", help="no plotting")

args = parser.parse_args()
alg = args.algorithm
test = args.test
pyro.set_rng_seed(args.seed)
nsamples = args.nsamples
assert nsamples >= 800
noplot = args.noplot

O111_mean = [6., 4., 0]
O111_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O111_shape = Polygon([[0, 0], [1, 0], [.5, .866]])
O111 = (O111_mean, O111_cov, O111_shape)

O121_mean = [5.5, 6, 0]
O121_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O121_shape = Polygon([[0, 0], [2, 0], [2, 1], [0, 1]])
O121 = (O121_mean, O121_cov, O121_shape)

R11 = [O111,]
R12 = [O121,]

S1 = ([0.8, 0.2], [R11, R12])

O211_mean = [2., 3, 0]
O211_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O211_shape = Polygon([[0, 0], [1, 0], [1, .5], [0.5, 0.25], [0, 1]])
O211 = (O211_mean, O211_cov, O211_shape)

O212_mean = [3.6, 4.6, 0]
O212_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O212_shape = Polygon([[0, 0], [.5, 0], [.5, .5], [0, .5]])
O212 = (O212_mean, O212_cov, O212_shape)

O221_mean = [5., 8., 0]
O221_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O221_shape = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
O221 = (O221_mean, O221_cov, O221_shape)

R21 = [O211, O212,]
R22 = [O221,]

S2 = ([.2, .8], [R21, R22])

WORLD = [S1, S2]
COLORS = [["#2980b9", "#7f5bb5"], ["#e67e22", "#e6bf22"]]
#[[blue,purple], [orange,yellow]]

X_MAX = 10.
Y_MAX = 10.
GOAL_X_MIN = 1.8
GOAL_X_MAX = 2.2
GOAL_Y_MIN = 4.8
GOAL_Y_MAX = 5.2
GOAL_THETA_MIN = math.pi * 0.
GOAL_THETA_MAX = math.pi * 2.
START_X = 8
START_Y = 4
START_THETA = math.pi * 7 / 10
ROBOT_WIDTH = 0.628
ROBOT_HEIGHT = 0.30
VEHICLE_SHAPE = Polygon(
    [[0, 0], [ROBOT_WIDTH, 0], [ROBOT_WIDTH, ROBOT_HEIGHT], [0, ROBOT_HEIGHT]])
INIT_FOOTPRINT = Footprint(VEHICLE_SHAPE, START_X, START_Y, START_THETA)

DELTA_T = .6
VELOCITY = .8
MAX_COLLISION_PROB = 0.01

SCENE = Scene(X_MAX, Y_MAX, WORLD, nsamples, COLORS)
MOTION_PRIM = get_motion_primitives_diff_drive(
    0, 0, 0, VELOCITY, DELTA_T)
GOAL = Goal(GOAL_X_MIN, GOAL_X_MAX, GOAL_Y_MIN,
            GOAL_Y_MAX, GOAL_THETA_MIN, GOAL_THETA_MAX)

print("Max collision prob = {}".format(MAX_COLLISION_PROB))

if args.nodisc:
    INIT_STATE = problem.State(INIT_FOOTPRINT, START_X, START_Y, START_THETA)
    PROBLEM = problem.Problem(INIT_STATE, GOAL, SCENE, MOTION_PRIM,
                              DELTA_T, VELOCITY, VEHICLE_SHAPE, MAX_COLLISION_PROB, test)
else:
    INIT_STATE = disc_state_problem.State(
        INIT_FOOTPRINT, START_X, START_Y, START_THETA)
    PROBLEM = disc_state_problem.Problem(
        INIT_STATE, GOAL, SCENE, MOTION_PRIM, DELTA_T, VELOCITY, VEHICLE_SHAPE, MAX_COLLISION_PROB, test)

if args.algorithm == 0:
    SEARCH = lambda: astar_search(PROBLEM, n_saved_explored_states=3000)
elif args.algorithm == 1:
    SEARCH = lambda: rrt_search(PROBLEM, goal_bias=0.3, max_iter=3000, n_saved_explored_states=3000)
else:
    assert(False)

start_time = time.time()

last_node, nexplored, explored_states = SEARCH()

end_time = time.time()
print("Elapsed time: " + str(end_time - start_time))

num_nominal_coll_checks = 0
num_actual_coll_checks = 0
for world in SCENE.worlds:
    num_nominal_coll_checks += world.num_nominal_coll_checks
    num_actual_coll_checks += world.num_actual_coll_checks
print("Nominal collision checks: " + str(num_nominal_coll_checks))
print("Actual collision checks:  " + str(num_actual_coll_checks))

def true_CP(scene, last_state):
    n_collisions = 0
    for w in scene.worlds:
        collides = problem.world_collides_state(w, last_state)
        if collides:
            n_collisions += 1
    cp = n_collisions / len(scene.worlds)
    return cp

if last_node is not None:
    solution = last_node.path()

    cp_stale = true_CP(SCENE, last_node.state)
    print("True CP = {} (Computed with stale samples)".format(cp_stale))

    SCENE_FRESH = Scene(X_MAX, Y_MAX, WORLD, nsamples, COLORS)
    cp_fresh = true_CP(SCENE_FRESH, last_node.state)
    print("True CP = {} (Computed with fresh samples)".format(cp_fresh))
else:
    print("Failed to find a solution")
    solution = None

if not noplot:
    SCENE.plot()

    plot_solution(PROBLEM, solution, explored_states=explored_states)

    plt.gca().set_aspect('equal', 'datalim')
    plt.show(block=True)
