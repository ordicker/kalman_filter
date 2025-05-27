import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.spatial.transform import Rotation as R

def predict(x, accel, gyro, dt):
    """
    x[0:3] = x, y ,z
    x[3:6] = vx, vy, vz
    x[6:9] = \phi, \theta, \psi
    x[9:12] = bias accel_{x, y, z}
    x[12:15] = bias omega_{\phi, \theta, \psi}
    """
    x_ = jnp.zeros_like(x)
    acc_bias = x[9:12]
    gyro_bias = x[12:15]
    
    acc_unbiased = accel - acc_bias
    gyro_unbiased = gyro - gyro_bias
    
    delta_theta = gyro_unbiased * dt
    head = x[6:9]+delta_theta
    att = R.from_rotvec(head)
    acc_ned = att.apply(acc_unbiased, inverse=True)
    gravity = np.array([0, 0, 9.81])
    acc_ned -= gravity
    acc_ned = att.apply(acc_ned)

    x_ = x_.at[3:6].set(x_[3:6]+acc_ned*dt)
    x_ = x_.at[0:3].set(x_[0:3]+x_[3:6]*dt)
    x_ = x_.at[6:9].set(head)
    x_ = x_.at[9:].set(x[9:])
    return x_


def h(x):
    return x[0:3]

def kalman_gain(P, h, x, R):
    H = jax.jacobian(h)(x)
    inv = jnp.linalg.inv(R+H@P@H.T)
    return P@H.T@inv

def error_propagation(P,f,Q):
    F = jax.jacobian(f)(x)
    return F@P@F.T+Q

def meas_update(K, h, x, z):
    H = jax.jacobian(h)(x)
    return x + K@(z-h(x)), (jnp.eye(15) - K@H)@P

###############################################################################
#                              simulations params                             #
###############################################################################
dt = 0.01
T = 30
N = int(T / dt)
gps_update_interval = int(1 / dt)

# Circle parameters
R_circ = 100
omega = 2 * jnp.pi / T
bias_accel = jnp.array([0.1, -0.05, 0.02])
bias_gyro = jnp.array([0.005, 0.002, -0.001])
R_gps = jnp.diag(jnp.array([3.1, 3.1, 6.0]))
Q = jnp.diag(jnp.array(
            [1e-0, 1e-0, 1e-0, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]
            + [0.0] * 3))

x = jnp.zeros(15)
P = jnp.eye(15)*1e2

###############################################################################
#                                  simulation                                 #
###############################################################################
import numpy as np

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
    acc_true = np.array(
        [-R_circ * omega**2 * np.cos(angle), -R_circ * omega**2 * np.sin(angle), 0]
    )
    gyro_true = np.array([0, 0, omega])  # rotating in Z

    R_b2n = R.from_euler("z", angle).as_matrix()
    acc_body = R.from_matrix(R_b2n).inv().apply(acc_true + np.array([0, 0, -9.81]))
    gyro_body = R.from_matrix(R_b2n).inv().apply(gyro_true)

    # imu_acc = acc_body + bias_accel
    # imu_gyro = gyro_body + bias_gyro

    imu_acc = acc_body
    imu_gyro = gyro_body

    # kalman filter block
    f = lambda x: predict(x, imu_acc, imu_gyro, dt)
    x = f(x)
    P = error_propagation(P,f,Q)
    
    if i % gps_update_interval == 0:
        gps_noise = np.random.normal(0, [3.1, 3.1, 6.0])
        # gps_noise = np.random.normal(0, [0.01, 0.01, 0.01])
        gps_meas = pos_true + gps_noise
        K = kalman_gain(P, h, x, R_gps)
        x, P = meas_update(K, h, x, gps_meas)


    
    pos_est = x[:3].copy()
    acc_b, gyro_b = x[9:12].copy(), x[12:15].copy()

    pos_log.append(pos_est.flatten())
    pos_true_log.append(pos_true)
    bias_acc_log.append(acc_b)
    bias_gyro_log.append(gyro_b)
    pos_var = np.diag(P)[0:3]  # variance of position x/y/z
    pos_var_log.append(pos_var)




###############################################################################
#                                  plotting!                                  #
###############################################################################

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
