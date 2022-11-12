import numpy as np
import torch
import torch.nn as nn


class Sine(nn.Module):
    """Sine activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


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
        n = out.size(1)

        return torch.tensor_split(out, n, dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")


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
        n = out.size(1)

        return torch.tensor_split(out, n, dim=1)

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
        # self.scaler.fit(X)
        # X = self.scaler.transform(X)

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

        return self.last(X)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")

        return
