import time
import os
from copy import deepcopy
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from aima.search import astar_search
from rrt_search import rrt_search

from problem import Goal, world_collides_state
import problem
import disc_state_problem

from motion import get_motion_primitives_diff_drive
from plot import plot_solution

import pyro


class RobotSetting:
    def __init__(self, init_footprint, start_x, start_y, start_theta, goal_x_min, goal_x_max, goal_y_min, goal_y_max,
                 goal_theta_min, goal_theta_max):
        self.init_footprint = init_footprint
        self.start_x = start_x
        self.start_y = start_y
        self.start_theta = start_theta

        self.goal = Goal(goal_x_min, goal_x_max, goal_y_min, goal_y_max, goal_theta_min, goal_theta_max)
        self.delta_t = 0.6
        self.velocity = 0.8

        self.motion_prim = get_motion_primitives_diff_drive(0, 0, 0, self.velocity, self.delta_t)

        robot_width = 0.628
        robot_height = 0.30
        self.vehicle_shape = Polygon([[0, 0], [robot_width, 0], [robot_width, robot_height], [0, robot_height]])


class Experiment:
    def __init__(self, exp_type, scenes, robot_setting, max_collision_prob, search_algorithm_st, sampling_algorithm, max_iter,
                 repetitions):
        self.exp_type = exp_type
        self.scenes = scenes
        self.robot_setting = robot_setting
        self.max_collision_prob = max_collision_prob
        self.search_algorithm = search_algorithm_st
        self.sampling_algorithm = sampling_algorithm
        self.max_iter = max_iter
        self.repetitions = repetitions

    def compute_true_CP(self, last_state, n_samples):
        print("computing true cp....")
        n_collisions = 0
        scene = deepcopy(self.scenes['truep'])
        witer = scene.worlds_iter()
        sample_collision = lambda: world_collides_state(next(witer), last_state)

        for _ in range(n_samples):

            collides = sample_collision()
            if collides:
                n_collisions += 1

        return n_collisions / n_samples

    def run(self):
        logdirname = self.exp_type
        os.system("rm -R ../log/" + logdirname)
        os.system("mkdir -p ../log/" + logdirname)
        fig_ploted = False
        pyro.set_rng_seed(101)
        for i in range(self.repetitions):
            print("NOW RUNNING EXPERIMENT " + str(i))
            scene_copy = deepcopy(self.scenes[i])

            if self.search_algorithm == 'astar':
                init_state = disc_state_problem.DiscState(self.robot_setting.init_footprint, self.robot_setting.start_x,
                                                      self.robot_setting.start_y, self.robot_setting.start_theta)

                problem_instance = disc_state_problem.Problem(
                    init_state, self.robot_setting.goal, scene_copy, self.robot_setting.motion_prim,
                    self.robot_setting.delta_t, self.robot_setting.velocity, self.robot_setting.vehicle_shape,
                    self.max_collision_prob, self.sampling_algorithm)

                search = lambda: astar_search(problem_instance, n_saved_explored_states=0, max_iter=self.max_iter)
            elif self.search_algorithm == 'rrt':
                init_state = problem.State(self.robot_setting.init_footprint, self.robot_setting.start_x,
                                           self.robot_setting.start_y, self.robot_setting.start_theta)

                problem_instance = problem.Problem(
                    init_state, self.robot_setting.goal, scene_copy, self.robot_setting.motion_prim,
                    self.robot_setting.delta_t, self.robot_setting.velocity, self.robot_setting.vehicle_shape,
                    self.max_collision_prob, self.sampling_algorithm)
                search = lambda: rrt_search(problem_instance, goal_bias=0.3, max_iter=self.max_iter,
                                            n_saved_explored_states=0)

            else:
                raise NotImplementedError()

            start_time = time.time()

            last_node, nexplored, explored_states = search()

            end_time = time.time()

            path_cost = last_node.path()[-1].path_cost if last_node is not None else 0

            if last_node is None:
                true_cp = float('inf')
            else:
                true_cp = self.compute_true_CP(last_node.state, 10000)

            num_nominal_coll_checks = 0
            num_actual_coll_checks = 0
            for world in problem_instance.scene.worlds:
                num_nominal_coll_checks += world.num_nominal_coll_checks
                num_actual_coll_checks += world.num_actual_coll_checks

            f = open("../log/" + logdirname + "/results.log", 'a+')

            logstring = str(last_node is None) + " " + str(nexplored) + " " + str(path_cost) + " " + \
                        str(end_time - start_time) + " " + str(num_nominal_coll_checks) + " " + \
                        str(num_actual_coll_checks) + " " + str(true_cp)

            f.write(logstring + "\n")
            f.close()

            if not fig_ploted and last_node is not None:
                fig_ploted = True
                fig, ax = plt.subplots()
                problem_instance.scene.plot()
                plot_solution(problem_instance, last_node.path(), explored_states=[])
                fig.savefig("../log/" + logdirname + "/example.png", bbox_inches='tight')
