#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

np.random.seed(6)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class EKF15:
    def __init__(self):
        self.x = np.zeros((15, 1))  # error state
        self.P = np.eye(15) * 1e2
        self.Q = np.diag(
            [1e-0, 1e-0, 1e-0, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]
            + [0.0] * 3
        )
        self.R_gps = np.diag([3.1, 3.1, 6.0]) ** 2
        # self.R_gps = np.diag([0.01, 0.01, 0.01]) ** 2

        self.pos = np.zeros((3, 1))
        self.vel = np.zeros((3, 1))
        self.att = R.identity()
        # Added
        # self.acc_bias = np.zeros((3, 1))
        # self.gyro_bias = np.zeros((3, 1))

    def predict(self, accel, gyro, dt):
        acc_bias = self.x[9:12].reshape((3, 1))
        gyro_bias = self.x[12:15].reshape((3, 1))
        # acc_bias = self.acc_bias.reshape((3, 1))
        # gyro_bias = self.gyro_bias.reshape((3, 1))

        acc_unbiased = accel.reshape((3, 1)) - acc_bias
        gyro_unbiased = gyro.reshape((3, 1)) - gyro_bias

        delta_theta = gyro_unbiased * dt
        delta_rot = R.from_rotvec(delta_theta.flatten())
        self.att = self.att * delta_rot

        acc_ned = self.att.apply(acc_unbiased.flatten()).reshape((3, 1))
        gravity = np.array([[0], [0], [9.81]])
        acc_ned -= gravity

        self.vel += acc_ned * dt
        self.pos += self.vel * dt + 1 / 2 * acc_ned * dt**2

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew(acc_ned.flatten())
        F[3:6, 9:12] = -self.att.as_matrix()
        F[6:9, 12:15] = -np.eye(3)

        Phi = np.eye(15) + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Q * dt
        self.x = Phi @ self.x

    def update_gps(self, z):
        z = z.reshape((3, 1))
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)

        y = z - self.pos
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        self.x += dx
        self.P = (np.eye(15) - K @ H) @ self.P

        self.pos += dx[0:3]
        self.vel += dx[3:6]
        delta_theta = dx[6:9].flatten()
        delta_rot = R.from_rotvec(delta_theta)
        self.att = delta_rot * self.att

        # self.acc_bias += dx[9:12]
        # self.gyro_bias += dx[12:15]
        # self.x[0:9] = 0
        self.x[0:15] = 0

    def get_state(self):
        return self.pos.copy(), self.vel.copy(), self.att.as_quat()

    def get_bias(self):
        return self.x[9:12].flatten(), self.x[12:15].flatten()


# Simulation settings
dt = 0.01
T = 30
N = int(T / dt)
gps_update_interval = int(1 / dt)

# Circle parameters
R_circ = 100
omega = 2 * np.pi / T
bias_accel = np.array([0.1, -0.05, 0.02])
bias_gyro = np.array([0.005, 0.002, -0.001])

ekf = EKF15()
pos_log, pos_true_log = [], []
bias_acc_log, bias_gyro_log = [], []
pos_var_log = []

for i in range(N):
    t = i * dt
    angle = omega * t
    pos_true = np.array([R_circ * np.cos(angle), R_circ * np.sin(angle), -9.81 * t**2])
    # pos_true = np.array([R_circ * np.cos(angle), R_circ * np.sin(angle), 0])
    vel_true = np.array(
        [-R_circ * omega * np.sin(angle), R_circ * omega * np.cos(angle), -9.81 * t]
    )
    # vel_true = np.array(
    #    [-R_circ * omega * np.sin(angle), R_circ * omega * np.cos(angle), 0]
    # )
    acc_true = np.array(
        [-R_circ * omega**2 * np.cos(angle), -R_circ * omega**2 * np.sin(angle), 0]
    )
    gyro_true = np.array([0, 0, omega])  # rotating in Z

    R_b2n = R.from_euler("z", angle).as_matrix()
    acc_body = R.from_matrix(R_b2n).inv().apply(acc_true + np.array([0, 0, -9.81]))
    # acc_body = R.from_matrix(R_b2n).inv().apply(acc_true + np.array([0, 0, 0]))
    gyro_body = R.from_matrix(R_b2n).inv().apply(gyro_true)

    # imu_acc = acc_body + bias_accel
    # imu_gyro = gyro_body + bias_gyro

    imu_acc = acc_body
    imu_gyro = gyro_body

    ekf.predict(imu_acc, imu_gyro, dt)

    if i % gps_update_interval == 0:
        gps_noise = np.random.normal(0, [3.1, 3.1, 6.0])
        # gps_noise = np.random.normal(0, [0.01, 0.01, 0.01])
        gps_meas = pos_true + gps_noise
        ekf.update_gps(gps_meas)

    pos_est, _, _ = ekf.get_state()
    acc_b, gyro_b = ekf.get_bias()

    pos_log.append(pos_est.flatten())
    pos_true_log.append(pos_true)
    bias_acc_log.append(acc_b)
    bias_gyro_log.append(gyro_b)
    pos_var = np.diag(ekf.P)[0:3]  # variance of position x/y/z
    pos_var_log.append(pos_var)

# Convert to arrays
pos_log = np.array(pos_log)
pos_true_log = np.array(pos_true_log)
bias_acc_log = np.array(bias_acc_log)
bias_gyro_log = np.array(bias_gyro_log)
pos_var_log = np.array(pos_var_log)
time = np.linspace(0, T, len(pos_log))

# Plots
plt.figure()
plt.plot(pos_true_log[:, 1], pos_true_log[:, 0], label="True Trajectory")
plt.plot(pos_log[:, 1], pos_log[:, 0], "--", label="Estimated Trajectory")
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.axis("equal")
plt.title("Circular Trajectory: True vs Estimated")
plt.legend()
plt.grid(True)

# plt.figure()
# plt.plot(time, bias_acc_log[:, 0], label="Acc Bias X")
# plt.plot(time, bias_acc_log[:, 1], label="Acc Bias Y")
# plt.plot(time, bias_acc_log[:, 2], label="Acc Bias Z")
# plt.title("Accelerometer Bias Estimates")
# plt.xlabel("Time [s]")
# plt.ylabel("Bias [m/s²]")
# plt.legend()
# plt.grid(True)
#
# plt.figure()
# plt.plot(time, bias_gyro_log[:, 0], label="Gyro Bias X")
# plt.plot(time, bias_gyro_log[:, 1], label="Gyro Bias Y")
# plt.plot(time, bias_gyro_log[:, 2], label="Gyro Bias Z")
# plt.title("Gyroscope Bias Estimates")
# plt.xlabel("Time [s]")
# plt.ylabel("Bias [rad/s]")
# plt.legend()
# plt.grid(True)
###################################################################
###################################################################
###################################################################
# pos_log = np.array(pos_log)
# pos_true_log = np.array(pos_true_log)
# pos_var_log = np.array(pos_var_log)
# time = np.linspace(0, T, len(pos_log))
#
# labels = ["North", "East", "Down"]
# for i in range(3):
#    plt.figure()
#    plt.plot(time, pos_log[:, i], label=f"Estimated {labels[i]}")
#    plt.plot(time, pos_true_log[:, i], "--", label=f"True {labels[i]}")
#    std = np.sqrt(pos_var_log[:, i])
#    plt.fill_between(
#        time,
#        pos_log[:, i] - std,
#        pos_log[:, i] + std,
#        color="gray",
#        alpha=0.3,
#        label="±1σ",
#    )
#    plt.xlabel("Time [s]")
#    plt.ylabel(f"{labels[i]} Position [m]")
#    plt.title(f"{labels[i]} Position with Uncertainty")
#    plt.legend()
#    plt.grid(True)
###################################################################
###################################################################
###################################################################

pos_log = np.array(pos_log)
pos_true_log = np.array(pos_true_log)
pos_var_log = np.array(pos_var_log)
time = np.linspace(0, T, len(pos_log))

labels = ["North", "East", "Down"]
for i in range(3):
    plt.figure()
    plt.plot(time, pos_log[:, i] - pos_true_log[:, i], label=f"Error")
    std = np.sqrt(pos_var_log[:, i])
    plt.fill_between(
        time,
        -std,
        +std,
        color="gray",
        alpha=0.3,
        label="±1σ",
    )
    plt.xlabel("Time [s]")
    plt.ylabel(f"{labels[i]} Position [m]")
    plt.title(f"{labels[i]} Position with Uncertainty")
    plt.legend()
    plt.grid(True)

plt.show()
