"""
This file contains functions for retrieving motion primitives for different robot types. Each motion primitive is
specified by the new x, y, and theta locations.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

CONSTANT_ANGULAR_SPEEDS = list(map(lambda x: math.radians(
    x), [0, 22.5, 45, 67.5, 90, -22.5, -45, -67.5, -90]))


def get_motion_primitives_diff_drive(x, y, theta, v, dt, small_dt=0.1):
    """
    This function returns the motion primitives of a differential drive robot (in terms of configurations reached)
    by applying fixed linear v and angular speeds above from x,y, theta for dt time. Small_dt can be used to specify
    intermediate configurations that are reached along the trajectory.
    """
    motion_primitives = []

    for w_n in CONSTANT_ANGULAR_SPEEDS:
        w = w_n
        def fun(t, y):

            ydot = np.empty((3,))

            ydot[0] = v*math.cos(y[2])
            ydot[1] = v*math.sin(y[2])
            ydot[2] = w

            return ydot

        n_steps = int(dt/small_dt)
        motion_primitives.append([])
        for i in range(n_steps):
            dt = (i + 1)*small_dt
            sol = solve_ivp(fun, (0, dt), np.array([x, y, theta]))
            x_sol = sol.y[0][len(sol.t) - 1]
            y_sol = sol.y[1][len(sol.t) - 1]
            theta_sol = sol.y[2][len(sol.t) - 1]
            motion_primitives[-1].append([x_sol, y_sol, theta_sol])

    return motion_primitives

def get_new_configuration_from_primitive(x, y, theta, delta_a, delta_b, delta_theta):
    x_new = x + math.cos(theta) * delta_a - math.sin(theta)* delta_b
    y_new = y + math.sin(theta) * delta_a + math.cos(theta)* delta_b
    theta_new = theta + delta_theta
    return (x_new, y_new, theta_new)

def get_new_configurations_from_primitives(x, y, theta, motion_primitives):
    """
    During the search, primitives are not recomputed from scratch. They are all stored w.r.t. a given reference frame.
    This function returns the correct new robots' positions from an arbitrary configuration, after having applied each
    primitive in motion_primitives (computed e.g. from the origin)
    """
    new_configs = []

    for m_p in motion_primitives:
        f_m_p = m_p[-1]
        new_config = get_new_configuration_from_primitive(x, y, theta, f_m_p[0], f_m_p[1], f_m_p[2])
        new_configs.append(new_config)

    return new_configs

@DeprecationWarning
def get_intermediate_configurations_from_primitive_sequence(x, y, theta, motion_primitive):
    """
    Same as above, but with intermediate configurations as well.
    """
    new_configs = []

    for m_p in motion_primitive:
        new_config = get_new_configuration_from_primitive(x, y, theta, m_p[0], m_p[1], m_p[2])
        new_configs.append(new_config)

    return new_configs
