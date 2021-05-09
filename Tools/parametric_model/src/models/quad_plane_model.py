__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """


import numpy as np
import math

from .dynamics_model import DynamicsModel
from .rotor_models import RotorModel
from sklearn.linear_model import LinearRegression
from scipy.linalg import block_diag
from .model_plots import model_plots, quad_plane_model_plots
from .model_config import ModelConfig
from .aerodynamic_models import AeroModelAAE


"""This model estimates forces and moments for quad plane as for example the standard vtol in gazebo."""


class QuadPlaneModel(DynamicsModel):
    def __init__(self, rel_data_path, config_file="qpm_gazebo_standard_vtol_config.yaml"):
        self.config = ModelConfig(config_file)
        super(QuadPlaneModel, self).__init__(
            config_dict=self.config.dynamics_model_config, rel_data_path=rel_data_path)

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]

    def prepare_regression_matrices(self):

        if "V_air_body_x" not in self.data_df:
            self.normalize_actuators()
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        # Rotor features
        angular_vel_mat = self.data_df[[
            "ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy()
        self.compute_rotor_features(self.rotor_config_dict, angular_vel_mat)

        if (self.estimate_forces):
            # Aerodynamics features
            airspeed_mat = self.data_df[["V_air_body_x",
                                        "V_air_body_y", "V_air_body_z"]].to_numpy()
            flap_commands = self.data_df[["u5", "u6", "u7"]].to_numpy()
            aoa_mat = self.data_df[["AoA"]].to_numpy()
            aero_model = AeroModelAAE(
                stall_angle=self.stall_angle, sig_scale_fac=self.sig_scale_fac)
            X_aero_forces, aero_coef_list = aero_model.compute_aero_features(
                airspeed_mat, aoa_mat, flap_commands)
            self.X_forces = np.hstack((self.X_rotor_forces, X_aero_forces))
            X = self.X_forces

            # Accelerations
            accel_body_mat = self.data_df[[
                "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()
            self.y_forces = accel_body_mat.flatten()
            y = self.y_forces

            # Set coefficients
            self.coef_name_list.extend(
                self.rotor_forces_coef_list + aero_coef_list)

        if (self.estimate_moments):
            # features due to rotation of body frame
            X_body_rot_moment, X_body_rot_moment_coef_list = self.compute_body_rotation_features(
                ["ang_vel_x", "ang_vel_y", "ang_vel_z"])
            self.X_moments = np.hstack(
                (self.X_rotor_moments, X_body_rot_moment))
            X = self.X_moments

            # Angular acceleration
            angular_accel_body_mat = self.data_df[[
                "ang_acc_x", "ang_acc_y", "ang_acc_z"]].to_numpy()
            self.y_moments = angular_accel_body_mat.flatten()
            y = self.y_moments

            # Set coefficients
            self.coef_name_list.extend(
                self.rotor_moments_coef_list + X_body_rot_moment_coef_list)

        if (self.estimate_forces and self.estimate_moments):
            X = block_diag(self.X_forces, self.X_moments)
            y = np.hstack((self.y_forces, self.y_moments))

            # define separate features for plotting and predicitons
            self.X_forces = X[0:self.X_forces.shape[0], :]
            self.X_moments = X[self.X_forces.shape[0]:X.shape[0], :]

        return X, y

    def estimate_model(self):
        print("Estimating quad plane model using the following data:")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        X, y = self.prepare_regression_matrices()

        self.reg = LinearRegression(fit_intercept=False)
        self.reg.fit(X, y)

        print("regression complete")
        metrics_dict = {"R2": float(self.reg.score(X, y))}
        self.coef_name_list.extend(["intercept"])
        coef_list = list(self.reg.coef_) + [self.reg.intercept_]
        self.generate_model_dict(coef_list, metrics_dict)
        self.save_result_dict_to_yaml(file_name="quad_plane_model")

        return

    def plot_model_predicitons(self):

        y_forces_pred = self.reg.predict(self.X_forces)
        y_moments_pred = self.reg.predict(self.X_moments)

        model_plots.plot_accel_predeictions(
            self.y_forces, y_forces_pred, self.data_df["timestamp"])
        model_plots.plot_angular_accel_predeictions(
            self.y_moments, y_moments_pred, self.data_df["timestamp"])
        model_plots.plot_az_and_collective_input(
            self.y_forces, y_forces_pred, self.data_df[["u0", "u1", "u2", "u3"]],  self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_z_direction(
            self.y_forces, y_forces_pred, self.data_df["V_air_body_z"], self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_y_direction(
            self.y_forces, y_forces_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])
        quad_plane_model_plots.plot_accel_predeictions_with_flap_outputs(
            self.y_forces, y_forces_pred, self.data_df[["u5", "u6", "u7"]], self.data_df["timestamp"])
        return
