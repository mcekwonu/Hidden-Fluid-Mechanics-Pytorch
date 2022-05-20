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


class Sine(nn.Module):
    """Sine activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


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


class Neural_Net(nn.Module):
    """A simple multilayer perceptron (MLP) neural network.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list,
                 activation_name="sine",
                 init_method="xavier_normal"):
        super().__init__()

        activation_dict = {"sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU()}
        self.act = activation_name
        self.init_method = init_method
        self.num_layers = len(layers_list) - 1

        self.base = nn.Sequential()
        for i in range(0, self.num_layers - 1):
            self.base.add_module(
                f"{i}  Linear", nn.Linear(layers_list[i], layers_list[i + 1])
            )
            self.base.add_module(f"{i} Act_fn", activation_dict[self.act])
        self.base.add_module(
            f"{self.num_layers - 1}  Linear",
            nn.Linear(layers_list[self.num_layers - 1],
                      layers_list[self.num_layers])
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.base.named_parameters():
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented yet!")

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1).requires_grad_(True)
        out = self.base(X)
        
        return torch.tensor_split(out, out.size(1), dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")
        
        return 

class ResidualBlock(nn.Module):
    """Residual block class for the Residual Network"""

    def __init__(self, in_dim, hidden_dim, out_dim, activation_name="sine"):
        super().__init__()

        activation_dict = {"sine": Sine(),
                           "tanh": nn.Tanh(),
                           "swish": nn.SiLU()}

        self.act = activation_name
        self.block = nn.Sequential()
        self.block.add_module("Act 0", activation_dict[self.act])
        self.block.add_module("Linear 0", nn.Linear(in_dim, hidden_dim))
        self.block.add_module("Act 1", activation_dict[self.act])
        self.block.add_module("Linear 1", nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        identity = x
        out = self.block(x)

        return out + identity


class ResNet(nn.Module):
    """MLP with residual connections.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list,
                 activation_name="sine", init_method="xavier_normal"):
        super().__init__()

        activation_dict = {
            "sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU()
        }
        self.init_method = init_method
        self.num_res_blocks = len(layers_list) - 2
        self.blocks = nn.Sequential()
        self.blocks.add_module("Linear 0",
                               nn.Linear(layers_list[0], layers_list[1])
                               )
        for i in range(self.num_res_blocks):
            res_blocks = ResidualBlock(layers_list[1],
                                       layers_list[1],
                                       layers_list[1],
                                       activation_name=activation_name)
            self.blocks.add_module(f"ResBlock {i + 1}", res_blocks)
        self.blocks.add_module("Final_Act", activation_dict[activation_name])
        self.blocks.add_module(
            "Linear_last", nn.Linear(layers_list[-2], layers_list[-1])
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.blocks.named_parameters():
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented!")

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1).requires_grad_(True)
        out = self.blocks(X)
        
        return torch.tensor_split(out, out.size(1), dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")
        
        return


class DenseResNet(nn.Module):
    """Dense Residual Neural network class with Fourier features, to enable multilayer perceptron
    (MLP) to learn high-frequency functions in low-dimensional problem domains.

    References:
    1. M. Tancik, P.P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal,
        R. Ramamoorthi, J.T. Barron and R. Ng, "Fourier Features Let Networks Learn High
        Frequency Functions in Low Dimensional Domains", NeurIPS, 2020.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        num_res_blocks (int): Number of residual network blocks. Default=5
        num_layers_per_block (int): Number of layers per block. Default=2
        fourier_features (bool): If to use fourier features. Default is True
        tune_beta (bool):
        m_freqs (int): fourier frequency. Default = 100
        sigma (int): std value for tuning fourier features. Default = 10
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list, num_res_blocks=5, num_layers_per_block=2,
                 activation_name="sine", init_method="xavier_normal",
                 fourier_features=True, tune_beta=True, m_freqs=100, sigma=10):
        super().__init__()

        activation_dict = {
            "sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU()
        }
        self.layers_list = layers_list
        self.num_res_blocks = num_res_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation_dict[activation_name]
        self.fourier_features = fourier_features
        self.init_method = init_method
        self.tune_beta = tune_beta
        # self.scaler = TorchMinMaxScaler()

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_res_blocks,
                                                self.num_layers_per_block))

        self.first = nn.Linear(layers_list[0], layers_list[1])
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(layers_list[1], layers_list[1])
                           for _ in range(num_layers_per_block)])
            for _ in range(num_res_blocks)])
        self.last = nn.Linear(layers_list[1], layers_list[-1])

        if fourier_features:
            self.first = nn.Linear(2 * m_freqs, layers_list[1])
            self.B = nn.Parameter(sigma * torch.randn(layers_list[0], m_freqs))

        self.init_weights()

    def init_weights(self):
        for name, param in (self.first.named_parameters() and
                            self.resblocks.named_parameters() and
                            self.last.named_parameters()):
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented!")

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1).requires_grad_(True)
      
        if self.fourier_features:
            cosx = torch.cos(torch.matmul(X, self.B))
            sinx = torch.sin(torch.matmul(X, self.B))
            X = torch.cat((cosx, sinx), dim=1)
            X = self.activation(self.beta0 * self.first(X))
        else:
            X = self.activation(self.beta0 * self.first(X))

        for i in range(self.num_res_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](X))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
                X = z + X
        out = self.last(X)
        
        return torch.tensor_split(out, out.size(1), dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")
        
        return
