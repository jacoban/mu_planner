import math
from datetime import date

from shapely.geometry import Polygon

from footprint import Footprint
from scene import Scene

from experiment import RobotSetting, Experiment

import pyro

# also sets normal random seed
pyro.set_rng_seed(101)

SCENE_NAME = date.today().strftime("%d-%m-%Y") + '_scene_2'

### ROBOT SETTING

START_X = 5
START_Y = 0.2
START_THETA = math.pi/2.0
ROBOT_WIDTH = 0.628
ROBOT_HEIGHT = 0.30
VEHICLE_SHAPE = Polygon(
    [[0, 0], [ROBOT_WIDTH, 0], [ROBOT_WIDTH, ROBOT_HEIGHT], [0, ROBOT_HEIGHT]])
INIT_FOOTPRINT = Footprint(VEHICLE_SHAPE, START_X, START_Y, START_THETA)

GOAL_X_MIN = 4.5
GOAL_X_MAX = 5.5
GOAL_Y_MIN = 8
GOAL_Y_MAX = 9
GOAL_THETA_MIN = math.pi * 0.
GOAL_THETA_MAX = math.pi * 2.

robot_setting = RobotSetting(INIT_FOOTPRINT, START_X, START_Y, START_THETA, GOAL_X_MIN, GOAL_X_MAX, GOAL_Y_MIN,
                             GOAL_Y_MAX, GOAL_THETA_MIN, GOAL_THETA_MAX)

### END ROBOT SETTING

### SCENE
# first hyp
O111_mean = [1., 2.5, 0]
O111_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O111_shape = Polygon([[0, 0], [4.5, 0], [4.5, 1], [0, 1]])
O111 = (O111_mean, O111_cov, O111_shape)

O112_mean = [3., 4.5, 0]
O112_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O112_shape = Polygon([[0, 0], [4.5, 0], [4.5, 1], [0, 1]])
O112 = (O112_mean, O112_cov, O112_shape)

# second hyp
O121_mean = [1., 4, 0]
O121_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O121_shape = Polygon([[0, 0], [4.5, 0], [4.5, 1], [0, 1]])
O121 = (O121_mean, O121_cov, O121_shape)

O122_mean = [3., 6, 0]
O122_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O122_shape = Polygon([[0, 0], [4.5, 0], [4.5, 1], [0, 1]])
O122 = (O122_mean, O122_cov, O122_shape)

R11 = [O111, O112]
R12 = [O121, O122]

S1 = ([0.01, 0.99], [R11, R12])

WORLD = [S1]
COLORS = [["orange", "red"]]

X_MAX = 10.
Y_MAX = 10.
NSAMPLES = 1000

SCENES = {}
for i in range(200):
    SCENES[i] = Scene(X_MAX, Y_MAX, WORLD, NSAMPLES, COLORS)

SCENES['truep'] = Scene(X_MAX, Y_MAX, WORLD, 10000, COLORS)
### END SCENE

for sampling_algorithm in ['sprt', 'mc']:
    for search_algorithm in ['rrt', 'astar']:
        collision_probs = [0.01, 0.1, 0.25]

        for max_collision_prob in collision_probs:
            experiment = Experiment("%s_%s_%s_%s" % (SCENE_NAME, search_algorithm, sampling_algorithm,
                                                     str(max_collision_prob)),
                                    SCENES, robot_setting, max_collision_prob, search_algorithm,
                                    sampling_algorithm, 5000, 200)

            experiment.run()
