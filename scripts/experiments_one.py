import math
from datetime import date

from shapely.geometry import Polygon

from footprint import Footprint
from scene import Scene

from experiment import RobotSetting, Experiment

import pyro

# just to be sure
pyro.set_rng_seed(101)

SCENE_NAME = date.today().strftime("%d-%m-%Y") + '_scene_1'

### ROBOT SETTING

START_X = 8
START_Y = 0.3
START_THETA = math.pi
ROBOT_WIDTH = 0.628
ROBOT_HEIGHT = 0.30
VEHICLE_SHAPE = Polygon(
    [[0, 0], [ROBOT_WIDTH, 0], [ROBOT_WIDTH, ROBOT_HEIGHT], [0, ROBOT_HEIGHT]])
INIT_FOOTPRINT = Footprint(VEHICLE_SHAPE, START_X, START_Y, START_THETA)

GOAL_X_MIN = 8
GOAL_X_MAX = 9
GOAL_Y_MIN = 8
GOAL_Y_MAX = 9
GOAL_THETA_MIN = math.pi * 0.
GOAL_THETA_MAX = math.pi * 2.

robot_setting = RobotSetting(INIT_FOOTPRINT, START_X, START_Y, START_THETA, GOAL_X_MIN, GOAL_X_MAX, GOAL_Y_MIN,
                             GOAL_Y_MAX, GOAL_THETA_MIN, GOAL_THETA_MAX)

### END ROBOT SETTING

### SCENE
O111_mean = [7., 2., 0]
O111_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O111_shape = Polygon([[0, 0], [3, 0], [3, 3], [0, 3]])
O111 = (O111_mean, O111_cov, O111_shape)

R11 = [O111]
R12 = []

S1 = ([1.0], [R11])

O211_mean = [6., 5.8, 0]
O211_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O211_shape = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
O211 = (O211_mean, O211_cov, O211_shape)

O221_mean = [6.0, 6.7, 0]
O221_cov = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.03]]
O221_shape = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
O221 = (O221_mean, O221_cov, O221_shape)

R21 = [O211]
R22 = [O221]

# S2 = ([1.], [R21])
S2 = ([0.3, 0.7], [R21, R22])

WORLD = [S1, S2]
COLORS = [["blue"], ["orange", "red"]]

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
        collision_probs = [0.25, 0.1, 0.01]

        for max_collision_prob in collision_probs:
            experiment = Experiment("%s_%s_%s_%s" % (SCENE_NAME, search_algorithm, sampling_algorithm,
                                                     str(max_collision_prob)),
                                    SCENES, robot_setting, max_collision_prob, search_algorithm,
                                    sampling_algorithm, 5000, 200)

            experiment.run()
