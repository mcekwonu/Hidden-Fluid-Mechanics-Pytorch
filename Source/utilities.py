import numpy as np
import scipy.io
import torch
import torch.nn as nn
from pyDOE import lhs
import matplotlib.pyplot as plt


def read_data(filepath):
    data = scipy.io.loadmat(filepath)

    U_star = data["u_star"]
    V_star = data["v_star"]
    X_star = data["x_star"]
    Y_star = data["y_star"]
    T_star = data["t"]

    X_star = X_star.reshape(-1, 1)
    Y_star = Y_star.reshape(-1, 1)

    Nx = X_star.shape[0]
    Nt = T_star.shape[0]

    X = np.tile(X_star, (1, Nt))
    Y = np.tile(Y_star, (1, Nt))
    T = np.tile(T_star, (1, Nx)).T
    U = U_star
    V = V_star

    x = X.ravel().reshape(-1, 1)
    y = Y.ravel().reshape(-1, 1)
    t = T.ravel().reshape(-1, 1)
    u = U.ravel().reshape(-1, 1)
    v = V.ravel().reshape(-1, 1)

    temp = np.concatenate((x, y, t, u, v), 1)
    minmax_value = np.empty((2, 5))
    minmax_value[0, :] = np.min(temp, axis=0)
    minmax_value[1, :] = np.max(temp, axis=0)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    minmax_value = torch.tensor(minmax_value, dtype=torch.float32)

    return x, y, t, u, v, minmax_value


def read_data_portion(filepath, percent):
    np.random.seed(11)
    data = scipy.io.loadmat(filepath)

    U_star = data["u_star"]
    V_star = data["v_star"]
    X_star = data["x_star"]
    Y_star = data["y_star"]
    T_star = data["t"]

    X_star = X_star.reshape(-1, 1)
    Y_star = Y_star.reshape(-1, 1)

    Nx = X_star.shape[0]
    Nt = T_star.shape[0]

    X = np.tile(X_star, (1, Nt))
    Y = np.tile(Y_star, (1, Nt))
    T = np.tile(T_star, (1, Nx)).T

    indices_t = np.random.choice(Nt, int(percent * Nt), replace=False)
    indices_t = np.sort(indices_t)

    t = T[:, indices_t].reshape(-1, 1)
    x = X[:, indices_t].reshape(-1, 1)
    y = Y[:, indices_t].reshape(-1, 1)
    u = U_star[:, indices_t].reshape(-1, 1)
    v = V_star[:, indices_t].reshape(-1, 1)

    temp = np.concatenate((x, y, t, u, v), 1)
    minmax_value = np.empty((2, 5))
    minmax_value[0, :] = np.min(temp, axis=0)
    minmax_value[1, :] = np.max(temp, axis=0)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    minmax_value = torch.tensor(minmax_value, dtype=torch.float32)

    return x, y, t, u, v, minmax_value


def compute_gradients(Y, x):
    dummy = torch.ones_like(Y, requires_grad=True)
    G = torch.autograd.grad(Y, x, grad_outputs=dummy, create_graph=True)[0]
    Y_x = torch.autograd.grad(G, dummy, grad_outputs=torch.ones_like(G), create_graph=True)[0]
    return Y_x


def generate_eqn_data(lower_bound, upper_bound, samples, num_points):
    eqn_points = lower_bound + (upper_bound - lower_bound) * lhs(samples, num_points)
    perm = np.random.permutation(eqn_points.shape[0])
    new_points = eqn_points[perm, :]
    return torch.from_numpy(new_points).float()


def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError("Unknown input type! Expected torch.Tensor or np.ndarray, but got {}".format(
            type(inputs))
        )


def gradient_velocity_2D(u, v, x, y):
    u_x = compute_gradients(u, x)
    v_x = compute_gradients(v, x)
    u_y = compute_gradients(u, y)
    v_y = compute_gradients(v, y)
    return u_x, v_x, u_y, v_y


def strain_rate_2D(u, v, x, y):
    u_x, v_x, u_y, v_y = gradient_velocity_2D(u, v, x, y)
    return u_x, 0.5 * (v_x + u_y), v_y


class TorchMinMaxScaler:
    """MinMax Scaler

    Transforms data to range [-1, 1]

    Returns:
        A tensor with scaled features
    """

    def __init__(self):
        self.x_max = None
        self.x_min = None

    def fit(self, x):
        self.x_max = x.max(dim=0, keepdim=True)[0]
        self.x_min = x.min(dim=0, keepdim=True)[0]

    def transform(self, x):
        x.sub_(self.x_min).div_(self.x_max - self.x_min)
        x.mul_(2).sub_(1)
        return x

    def inverse_transform(self, x):
        x.add_(1).div_(2)
        x.mul_(self.x_max - self.x_min).add_(self.x_min)
        return x

    fit_transform = transform
