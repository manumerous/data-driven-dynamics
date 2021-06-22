__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import matplotlib.pyplot as plt
import numpy as np
import math


def plot_thrust_prediction_and_underlying_data(rotor_coef_dict, rotor, force_proj_data):
    fig, ax = plt.subplots()
    u_vec = np.linspace(
        0, 1, num=(101))
    thrust_vec = np.zeros(101)
    for i in range(u_vec.shape[0]):
        thrust_vec[i] = rotor.air_density * rotor.prop_diameter**4 * \
            u_vec[i]**2 * rotor_coef_dict["rot_thrust_quad"]

    force_proj_data_coll = []
    u_data_coll = []
    for i in range(force_proj_data.shape[0]):
        if abs(rotor.v_air_parallel_abs[i] <= 0.5):
            force_proj_data_coll.append(force_proj_data[i])
            u_data_coll.append(rotor.actuator_input_vec[i])

    ax.plot(rotor.actuator_input_vec, force_proj_data, 'o',
            label="underlying data", color='grey', alpha=0.25)
    ax.plot(u_vec, thrust_vec, label="prediction")

    ax.set_title("Tail Rotor Force over actuator input")
    plt.legend()


def plot_rotor_trust_3d(rotor_coef_dict, rotor, tail=False):
    fig, ax = plt.subplots(1)
    u_vec = np.arange(0, 1, .01)
    if tail:
        ang_vel_rpm = (-1895*u_vec**2 + 10882*u_vec + 213)
    else:
        ang_vel_rpm = (-3058 * u_vec**2 + 9945*u_vec + 226)
    v_air_par_vec = np.arange(0, 20, .1)
    u_vec, v_air_par_vec = np.meshgrid(ang_vel_rpm, v_air_par_vec)
    ang_vel_vec = ang_vel_rpm*math.pi/30
    f_thrust_mat = rotor.air_density * rotor.prop_diameter**4 * \
        (ang_vel_vec**2 * rotor_coef_dict["rot_thrust_quad"] + ang_vel_vec *
         v_air_par_vec * rotor_coef_dict["rot_thrust_lin"] / rotor.prop_diameter)

    # for i in range(u_vec.shape[0]):
    #     f_trust_zero_airspeed = rotor.air_density * rotor.prop_diameter**4 * \
    #         u_vec[i]**2 * rotor_coef_dict["rot_thrust_quad"]
    #     f_thrust_mat[i, :] = np.ones(
    #         (1, v_air_par_vec.shape[0])) * f_trust_zero_airspeed - v_air_par_vec * rotor.air_density * rotor.prop_diameter**3 * \
    #         u_vec[i]**2 * rotor_coef_dict["rot_thrust_lin"]
    ax = plt.axes(projection='3d')
    ax.plot_surface(ang_vel_rpm, v_air_par_vec, f_thrust_mat, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title("Rotor Thrust Force")
    ax.set_xlabel("Rotor RPM [1/min] ")
    ax.set_ylabel("Inflow Airspeed [m/s]")
    ax.set_zlabel("Rotor Thrust [N]")
