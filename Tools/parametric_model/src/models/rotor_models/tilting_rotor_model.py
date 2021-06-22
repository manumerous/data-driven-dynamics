__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from . import ChangingAxisRotorModel
import numpy as np
import pandas as pd
import math
from progress.bar import Bar
from . import RotorModel
from scipy.spatial.transform import Rotation


class TiltingRotorModel(ChangingAxisRotorModel):

    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, tilt_actuator_vec, air_density=1.225, angular_vel_mat=None):
        self.tilt_axis = np.array(rotor_config_dict["tilt_axis"]).reshape(3, 1)
        self.max_tilt_angle = rotor_config_dict["max_tilt_angle_deg"]*math.pi/180.0
        self.tilt_actuator_vec = np.array(tilt_actuator_vec)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.compute_rotor_axis_mat()
        super(TiltingRotorModel, self).__init__(rotor_config_dict, actuator_input_vec,
                                                v_airspeed_mat, air_density=1.225, angular_vel_mat=None)

    def compute_rotor_axis_mat(self):
        self.rotor_axis_mat = np.zeros((self.n_timestamps, 3))
        for i in range(self.n_timestamps):
            # Active vector rotation around tilt axis:
            rotvec = self.tilt_axis.flatten() * self.max_tilt_angle * \
                self.tilt_actuator_vec[i]
            R_active_tilt = Rotation.from_rotvec(
                rotvec).as_matrix()
            curr_axis = (
                R_active_tilt @ self.rotor_axis).flatten()
            self.rotor_axis_mat[i, :] = curr_axis/np.linalg.norm(curr_axis)

    def compute_actuator_force_features(self, index, rotor_axis=None):
        """compute thrust model using a 2nd degree model of the normalized actuator outputs

        Inputs:
        actuator_input: actuator input between 0 and 1
        v_airspeed: airspeed velocity in body frame, numpoy array of shape (3,1)

        For the model explanation have a look at the PDF.
        """

        actuator_input = self.actuator_input_vec[index]
        v_air_parallel_abs = self.v_air_parallel_abs[index]
        v_airspeed_perpendicular_to_rotor_axis = \
            self.v_airspeed_perpendicular_to_rotor_axis[index, :].reshape(
                (3, 1))
        ang_vel = (-3058 * actuator_input**2 + 9945 *
                   actuator_input + 226)*math.pi/30
        # ang_vel = actuator_input

        if rotor_axis is None:
            rotor_axis = self.rotor_axis

        # Thrust force computation
        X_thrust = rotor_axis @ np.array(
            [[(v_air_parallel_abs*ang_vel/self.prop_diameter), ang_vel**2]]) * self.air_density * self.prop_diameter**4
        # Drag force computation
        if (np.linalg.norm(v_airspeed_perpendicular_to_rotor_axis) >= 0.05):
            X_drag = - v_airspeed_perpendicular_to_rotor_axis @ np.array(
                [[ang_vel]])
        else:
            X_drag = np.zeros((3, 1))

        X_forces = np.hstack((X_drag, X_thrust))

        return X_forces
