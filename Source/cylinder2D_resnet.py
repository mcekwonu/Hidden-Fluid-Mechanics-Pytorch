import os
import time
import warnings
import datetime
import imageio as iio
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utilities import *

torch.manual_seed(24)
np.random.seed(24)
torch.set_default_tensor_type(torch.FloatTensor)
warnings.filterwarnings("ignore")


class CylinderResNetHFM(nn.Module):
    """2D Hidden Fluid Mechanics class.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons.
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
        _data : denotes training data (x_data, y_data, t_data, u_data, v_data)
        minmax: ndarray of minimum and maximum value of all training data
        batch_size: batch size for training
        lamda: regularization parameter for tuning PDE equation loss. Default is 1.0
        epochs: Number of epochs
        sample_epoch: Epoch to display training results
        save_name: Save model name. If None, the model name is automatically created from the class.
    """

    def __init__(self, layers_list, activation_name="sine",
                 init_method="xavier_normal", nn_type="resnet", save_name=None, *,
                 x_data, y_data, t_data, u_data, v_data, Re,
                 minmax, lamda, epochs, sample_epoch, learning_rate):
        super().__init__()

        self.x_data = x_data
        self.y_data = y_data
        self.t_data = t_data
        self.u_data = u_data
        self.v_data = v_data
        self.Rex = Re
        self.lamda = lamda
        self.minmax = minmax
        self.num_epochs = epochs
        self.lr = learning_rate
        self.interval = sample_epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if save_name is None:
            self.model_name = self.__class__.__qualname__
        else:
            self.model_name = save_name

        if nn_type == "vanilla":
            self.net_uvp = Neural_Net(layers_list,
                                      activation_name=activation_name,
                                      init_method=init_method)
        elif nn_type == "resnet":
            self.net_uvp = Neural_Net(layers_list,
                                      activation_name=activation_name,
                                      init_method=init_method)
        elif nn_type == "denseresnet":
            self.net_uvp = DenseResNet(layers_list,
                                       num_res_blocks=5,
                                       num_layers_per_block=2,
                                       fourier_features=True,
                                       tune_beta=True,
                                       m_freqs=10,
                                       sigma=1,
                                       activation_name=activation_name,
                                       init_method=init_method)

        self.optimizer = optim.Adam(self.net_uvp.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)
        self.selected_time = t_data.numpy()[2][0]

        self.create_dir(f"../logs/{self.model_name}/checkpoint")
        self.create_dir(f"../logs/{self.model_name}/model")
        self.create_dir(f"../logs/{self.model_name}/results")

    def physics_constraints(self, x, y, t, Rex):
        uvp = self.net_uvp(x, y, t)

        u = uvp[:, 0].reshape(-1, 1)
        v = uvp[:, 1].reshape(-1, 1)
        p = uvp[:, 2].reshape(-1, 1)

        u_t = compute_gradients(u, t)
        u_x = compute_gradients(u, x)
        u_y = compute_gradients(u, y)
        v_t = compute_gradients(v, t)
        v_x = compute_gradients(v, x)
        v_y = compute_gradients(v, y)
        p_x = compute_gradients(p, x)
        p_y = compute_gradients(p, y)
        u_xx = compute_gradients(u_x, x)
        u_yy = compute_gradients(u_y, y)
        v_xx = compute_gradients(v_x, x)
        v_yy = compute_gradients(v_y, y)

        e1 = u_t + (u * u_x + v * u_y) + p_x - (1. / Rex) * (u_xx + u_yy)
        e2 = v_t + (u * v_x + v * v_y) + p_y - (1. / Rex) * (v_xx + v_yy)
        e3 = u_x + v_y
        omega = v_x - u_y

        return u, v, p, omega, e1, e2, e3

    def loss_fn(self, outputs, targets):

        return nn.MSELoss(reduction="mean")(outputs, targets)

    def data_loss(self, x, y, t, u, v):
        uvp_pred = self.net_uvp(x, y, t)

        u_pred = uvp_pred[:, 0].reshape(-1, 1)
        v_pred = uvp_pred[:, 1].reshape(-1, 1)

        u_loss = self.loss_fn(u_pred, u)
        v_loss = self.loss_fn(v_pred, v)

        return u_loss, v_loss

    def equation_loss(self, x, y, t):
        _, _, _, _, e1, e2, e3 = self.physics_constraints(x, y, t, self.Rex)
        f_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(self.device)
        loss_equation = self.loss_fn(e1, f_zeros) + self.loss_fn(e2, f_zeros) + self.loss_fn(e3, f_zeros)

        return loss_equation

    def _train(self):
        choice = input("Resume (y) or New (n): ")
        if choice == "n" or choice == "N":
            print("\nStarting new training ...")

            device = self.device
            pinn_net = self.net_uvp.to(device)
            model_name = self.model_name
            losses = np.empty((0, 6), dtype=float)
            N_eqn = 1000000
            dimensions = 3

            x_data, y_data, t_data, u_data, v_data, minmax = (self.x_data,
                                                              self.y_data,
                                                              self.t_data,
                                                              self.u_data,
                                                              self.v_data,
                                                              self.minmax)
            data_stack = torch.cat([x_data, y_data, t_data, u_data, v_data], dim=1).numpy()

            lb = np.array([minmax.numpy()[0, 0], minmax.numpy()[0, 1], minmax.numpy()[0, 2]])
            ub = np.array([minmax.numpy()[1, 0], minmax.numpy()[1, 1], minmax.numpy()[1, 2]])
            Eqn_points = generate_eqn_data(lb, ub, dimensions, N_eqn)

            X_shuffle = self.shuffle_batch(x_data, y_data, t_data, u_data, v_data)
            del minmax, x_data, y_data, t_data, u_data, v_data

            ratio = 0.05  # changed from 0.02
            batch_size_data = int(ratio * X_shuffle.shape[0])
            batch_size_eqn = int(ratio * Eqn_points.shape[0])
            batch_iter = int(X_shuffle.size(0) / batch_size_data)
            eqn_iter = int(Eqn_points.size(0) / batch_size_eqn)
            train_loss = []

            start_time = time.time()
            for epoch in range(self.num_epochs):
                for it in range(batch_iter):
                    self.optimizer.zero_grad()
                    if it < batch_iter:
                        x_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 0].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        y_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 1].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        t_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 2].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        u_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 3].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        v_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 4].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        x_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 0].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        y_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 1].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                        t_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 2].reshape(
                            -1, 1).clone().requires_grad_(True).to(device)
                    elif it == eqn_iter:
                        if X_shuffle[it * batch_size_data:, 0].reshape(-1, 1).shape[0] == 0:
                            continue
                        else:
                            x_data = X_shuffle[it * batch_size_data:, 0].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            y_data = X_shuffle[it * batch_size_data:, 1].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            t_data = X_shuffle[it * batch_size_data:, 2].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            u_data = X_shuffle[it * batch_size_data:, 3].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            v_data = X_shuffle[it * batch_size_data:, 4].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            x_eqn = Eqn_points[it * batch_size_eqn:, 0].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            y_eqn = Eqn_points[it * batch_size_eqn:, 1].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)
                            t_eqn = Eqn_points[it * batch_size_eqn:, 2].reshape(-1, 1).clone().requires_grad_(
                                True).to(device)

                    loss_u, loss_v = self.data_loss(x_data, y_data, t_data, u_data, v_data)
                    loss_eqn = self.equation_loss(x_eqn, y_eqn, t_eqn)

                    loss = loss_u + loss_v + self.lamda * loss_eqn
                    train_loss.append(to_numpy(loss))
                    loss.backward()
                    self.optimizer.step()

                    u_pred, v_pred, _, _, _, _, _ = self.physics_constraints(x_data,
                                                                             y_data,
                                                                             t_data,
                                                                             self.Rex)
                    error_u = self.relative_error(u_data, u_pred)
                    error_v = self.relative_error(v_data, v_pred)

                    with torch.autograd.no_grad():
                        loss_u = to_numpy(loss_u).reshape(1, 1)
                        loss_v = to_numpy(loss_v).reshape(1, 1)
                        loss_eq = to_numpy(loss_eqn).reshape(1, 1)
                        total_loss = to_numpy(loss).reshape(1, 1)
                        lr = self.optimizer.param_groups[0]["lr"]

                        err_u = to_numpy(error_u).reshape(1, 1)
                        err_v = to_numpy(error_v).reshape(1, 1)

                self.callback(epoch, total_loss, loss_u, loss_v, loss_eq, err_u, err_v, lr)
                all_losses = np.concatenate([total_loss, loss_u, loss_v, loss_eq, err_u, err_v], axis=1)
                losses = np.append(losses, all_losses, axis=0)
                loss_log = pd.DataFrame(losses)
                loss_log.to_csv(
                    f"../logs/{model_name}/results/losses.csv",
                    index=False,
                    header=["Loss", "Loss U", "Loss V", "Loss Eqns", "Rel_u", "Rel_v"]
                )
                del loss_log

                if (epoch + 1) % self.interval == 0:
                    state = {
                        "epoch": epoch + 1,
                        "state_dict": pinn_net.state_dict(),
                        "optimizer_dict": self.optimizer.state_dict()
                    }
                    self.save_checkpoint(
                        state, checkpoint_dir=f"../logs/{model_name}/checkpoint"
                    )

                if epoch % self.interval == 0:
                    self.exact_and_predict_at_selected_time(self.selected_time, data_stack)

            train_loss = np.array(train_loss).mean()
            self.scheduler.step(train_loss)

            elapsed = time.time() - start_time
            self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model")

            with open(f"../logs/{model_name}/results/training_metadata.txt", "w") as f:
                f.write(f"{model_name} training metadata generated at {datetime.datetime.now()}\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"Iterations: {batch_iter * self.num_epochs}\n")
                f.write(f"Training epochs: {self.num_epochs}\n")
                f.write(f"Training time: {elapsed / 3600:2.0f}h ({elapsed:.2f}s)\n")

            print(f"\nTraining completed in {elapsed / 3600:^2.0f}h")

        elif choice == "y" or choice == "Y":
            print("\nResume training from last saved checkpoint ...\n")
            self._resume()

    def _resume(self):
        checkpoint_path = f"../logs/{self.model_name}/checkpoint"
        restore_path = [os.path.join(checkpoint_path, f)
                        for f in sorted(os.listdir(checkpoint_path))
                        if f.endswith("pth")]
        checkpoint = torch.load(*restore_path)

        device = self.device
        model_name = self.model_name
        pinn_net = self.net_uvp.to(device)
        restore_epoch = checkpoint["epoch"]
        pinn_net.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
        losses = np.empty((0, 6), dtype=float)
        N_eqn = 1000000
        dimensions = 3

        x_data, y_data, t_data, u_data, v_data, minmax = (self.x_data,
                                                          self.y_data,
                                                          self.t_data,
                                                          self.u_data,
                                                          self.v_data,
                                                          self.minmax)
        data_stack = torch.cat([x_data, y_data, t_data, u_data, v_data], dim=1).numpy()

        lb = np.array([minmax.numpy()[0, 0], minmax.numpy()[0, 1], minmax.numpy()[0, 2]])
        ub = np.array([minmax.numpy()[1, 0], minmax.numpy()[1, 1], minmax.numpy()[1, 2]])
        Eqn_points = generate_eqn_data(lb, ub, dimensions, N_eqn)

        X_shuffle = self.shuffle_batch(x_data, y_data, t_data, u_data, v_data)
        del x_data, y_data, t_data, u_data, v_data, minmax

        ratio = 0.05  # changed from 0.02
        batch_size_data = int(ratio * X_shuffle.shape[0])
        batch_size_eqn = int(ratio * Eqn_points.shape[0])
        batch_iter = int(X_shuffle.size(0) / batch_size_data)
        eqn_iter = int(Eqn_points.size(0) / batch_size_eqn)
        train_loss = []

        if restore_epoch < self.num_epochs:
            restore_epoch = self.num_epochs
        else:
            restore_epoch += self.num_epochs

        start_time = time.time()

        for epoch in range(restore_epoch):
            for it in range(batch_iter):
                self.optimizer.zero_grad()
                if it < batch_iter:
                    x_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 0].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    y_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 1].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    t_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 2].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    u_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 3].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    v_data = X_shuffle[it * batch_size_data:(it + 1) * batch_size_data, 4].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    x_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 0].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    y_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 1].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                    t_eqn = Eqn_points[it * batch_size_eqn:(it + 1) * batch_size_eqn, 2].reshape(
                        -1, 1).clone().requires_grad_(True).to(device)
                elif it == eqn_iter:
                    if X_shuffle[it * batch_size_data:, 0].reshape(-1, 1).shape[0] == 0:
                        continue
                    else:
                        x_data = X_shuffle[it * batch_size_data:, 0].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        y_data = X_shuffle[it * batch_size_data:, 1].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        t_data = X_shuffle[it * batch_size_data:, 2].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        u_data = X_shuffle[it * batch_size_data:, 3].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        v_data = X_shuffle[it * batch_size_data:, 4].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        x_eqn = Eqn_points[it * batch_size_eqn:, 0].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        y_eqn = Eqn_points[it * batch_size_eqn:, 1].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)
                        t_eqn = Eqn_points[it * batch_size_eqn:, 2].reshape(-1, 1).clone().requires_grad_(
                            True).to(device)

                loss_u, loss_v = self.data_loss(x_data, y_data, t_data, u_data, v_data)
                loss_eqn = self.equation_loss(x_eqn, y_eqn, t_eqn)

                loss = loss_u + loss_v + self.lamda * loss_eqn
                train_loss.append(to_numpy(loss))
                loss.backward()

                self.optimizer.step()

                u_pred, v_pred, _, _, _, _, _ = self.physics_constraints(x_data, y_data, t_data, self.Rex)
                error_u = self.relative_error(u_data, u_pred)
                error_v = self.relative_error(v_data, v_pred)

                with torch.autograd.no_grad():
                    loss_u = to_numpy(loss_u).reshape(1, 1)
                    loss_v = to_numpy(loss_v).reshape(1, 1)
                    loss_eq = to_numpy(loss_eqn).reshape(1, 1)
                    total_loss = to_numpy(loss).reshape(1, 1)
                    lr = self.optimizer.param_groups[0]["lr"]

                    err_u = to_numpy(error_u).reshape(1, 1)
                    err_v = to_numpy(error_v).reshape(1, 1)

            self.callback(epoch, total_loss, loss_u, loss_v, loss_eq, err_u, err_v, lr)
            all_losses = np.concatenate([total_loss, loss_u, loss_v, loss_eq, err_u, err_v], axis=1)
            losses = np.append(losses, all_losses, axis=0)
            loss_log = pd.DataFrame(losses)
            loss_log.to_csv(f"../logs/{model_name}/results/losses.csv", index=False, header=False, mode="a")
            del loss_log

            if (epoch + 1) % self.interval == 0:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": pinn_net.state_dict(),
                    "optimizer_dict": self.optimizer.state_dict()
                }
                self.save_checkpoint(
                    state, checkpoint_dir=f"../logs/{model_name}/checkpoint"
                )

            if epoch % self.interval == 0:
                self.exact_and_predict_at_selected_time(self.selected_time, data_stack)

        train_loss = np.array(train_loss).mean()
        self.scheduler.step(train_loss)

        elapsed = time.time() - start_time
        self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model")

        with open(f"../logs/{model_name}/results/training_metadata.txt", "a") as f:
            f.write(f"Total epochs: {restore_epoch}\n")
            f.write(f"Total iterations: {batch_iter * restore_epoch}\n")
            f.write(f"Resume training time: {elapsed / 3600:2.0f}h ({elapsed:.2f}s)\n")

        print(f"\nTraining completed in {elapsed / 3600:^2.0f}h")

    def exact_and_predict_at_selected_time(self, selected_time, data):
        x = data[:, 0].copy().reshape(-1, 1)
        y = data[:, 1].copy().reshape(-1, 1)
        t = data[:, 2].copy().reshape(-1, 1)
        u = data[:, 3].copy().reshape(-1, 1)
        v = data[:, 4].copy().reshape(-1, 1)

        idx_time = np.where(t == selected_time)[0]
        x = np.unique(x).reshape(-1, 1)
        y = np.unique(y).reshape(-1, 1)
        mesh_x, mesh_y = np.meshgrid(x, y)
        u_selected = u[idx_time]
        v_selected = v[idx_time]
        x_flatten = mesh_x.flatten().reshape(-1, 1)
        y_flatten = mesh_y.flatten().reshape(-1, 1)
        t_flatten = np.ones_like(x_flatten) * selected_time

        x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(self.device)
        y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(self.device)
        t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(self.device)
        del x_flatten, y_flatten, t_flatten

        (u_pred,
         v_pred,
         p_pred,
         vort_pred,
         _,
         _,
         _) = self.physics_constraints(x_selected, y_selected, t_selected, self.Rex)

        u_pred = to_numpy(u_pred).reshape(mesh_x.shape)
        v_pred = to_numpy(v_pred).reshape(mesh_x.shape)
        p_pred = to_numpy(p_pred).reshape(mesh_x.shape)
        vort_pred = to_numpy(vort_pred).reshape(mesh_x.shape)
        u_exact = u_selected.reshape(mesh_x.shape)
        v_exact = v_selected.reshape(mesh_x.shape)

        self.plot_on_epoch_train(params_exact=u_exact, params_predict=u_pred, selected_time=selected_time, name="u")
        self.plot_on_epoch_train(params_exact=v_exact, params_predict=v_pred, selected_time=selected_time, name="v")
        self.plot_on_epoch_train(params_exact=None, params_predict=p_pred, selected_time=selected_time, name="p")
        self.plot_on_epoch_train(params_exact=None, params_predict=vort_pred, selected_time=selected_time,
                                 name="vorticity")
        plt.close("all")

    def predict_at_selected_time_with_error(self, data):

        x = data[:, 0].copy().reshape(-1, 1)
        y = data[:, 1].copy().reshape(-1, 1)
        t = data[:, 2].copy().reshape(-1, 1)
        u = data[:, 3].copy().reshape(-1, 1)
        v = data[:, 4].copy().reshape(-1, 1)
        min_data = np.min(data, axis=0).reshape(1, -1)
        max_data = np.max(data, axis=0).reshape(1, -1)

        x = np.unique(x).reshape(-1, 1)
        y = np.unique(y).reshape(-1, 1)
        mesh_x, mesh_y = np.meshgrid(x, y)

        start_time = t.min()
        end_time = t.max()
        time_interval = t[2] - t[1]

        n = int((end_time - start_time) / time_interval) + 1

        time_span = np.linspace(start_time, end_time, n)

        F_D, F_L = self.predict_drag_lift(t_cyl=t)

        for t_s in time_span:
            t_s = round(float(t_s), 3)
            idx_time = np.where(t == t_s)[0]
            u_snap = u[idx_time]
            v_snap = v[idx_time]
            x_flatten = mesh_x.ravel().reshape(-1, 1)
            y_flatten = mesh_y.ravel().reshape(-1, 1)
            t_flatten = np.ones_like(x_flatten) * t_s

            x_snap = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(self.device)
            y_snap = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(self.device)
            t_snap = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(self.device)

            (u_pred,
             v_pred,
             p_pred,
             vort_pred,
             _,
             _,
             _) = self.physics_constraints(x_snap, y_snap, t_snap, self.Rex)

            u_pred = to_numpy(u_pred).reshape(mesh_x.shape)
            v_pred = to_numpy(v_pred).reshape(mesh_x.shape)
            p_pred = to_numpy(p_pred).reshape(mesh_x.shape)
            vort_pred = to_numpy(vort_pred).reshape(mesh_x.shape)
            u_snap = u_snap.reshape(mesh_x.shape)
            v_snap = v_snap.reshape(mesh_x.shape)
            u_diff = (u_pred - u_snap) / u_snap
            v_diff = (v_pred - v_snap) / v_snap

            self.plot_exact_and_predict_in_time(u_snap, u_pred, t_s,
                                                min_value=min_data[0, 3],
                                                max_value=max_data[0, 3],
                                                name="u")
            self.plot_exact_and_predict_in_time(v_snap, v_pred, t_s,
                                                min_value=min_data[0, 4],
                                                max_value=max_data[0, 4],
                                                name="v")
            self.plot_relative_error_in_time(u_diff, t_s, name="u")
            self.plot_relative_error_in_time(v_diff, t_s, name="v")
            self.plot_predicted_params_in_time(p_pred, t_s, name="p")
            self.plot_predicted_params_in_time(vort_pred, t_s, name="vorticity")

            del u_snap, v_snap, x_flatten, y_flatten, t_flatten
            del x_snap, y_snap, t_snap
            plt.close("all")

        scipy.io.savemat(
            f"../logs/{self.model_name}/results/{self.model_name}_results_{time.strftime('%d_%m_%Y')}.mat",
            {"U_pred": u_pred, "V_pred": v_pred, "P_pred": p_pred, "Vorticity_pred": vort_pred,
             "F_D": F_D, "F_L": F_L}
        )

    def plot_relative_error_in_time(self, params, time_snap, name="param"):
        save_dir = f"../logs/{self.model_name}/snapshots/relative_error"
        self.create_dir(save_dir)

        min_value = np.min(params)
        max_value = np.max(params)
        v_norm = colors.Normalize(vmin=min_value, vmax=max_value)

        plt.figure(figsize=(8, 6))
        plt.imshow(params, cmap="jet", norm=v_norm)
        plt.title(f"Rel.error {name} (t = {time_snap:.2f}s)")
        plt.ylabel("y/D")
        plt.xlabel("x/D")
        plt.colorbar(shrink=0.8)
        plt.savefig(f"{save_dir}/{name}_{time_snap:.2f}.png")
        plt.close("all")

    def plot_predicted_params_in_time(self, params, time_select, name="param"):
        save_dir = f"../logs/{self.model_name}/snapshots/{name}"
        self.create_dir(save_dir)

        min_value = np.min(params)
        max_value = np.max(params)
        v_norm = colors.Normalize(vmin=min_value, vmax=max_value)

        plt.figure(figsize=(8, 6))
        plt.imshow(params, cmap="jet", norm=v_norm)
        plt.title(f"Learned {name}(x, y) (t = {time_select:.2f}s)")
        plt.ylabel("y/D")
        plt.xlabel("x/D")
        plt.colorbar(shrink=0.6)
        plt.savefig(f"{save_dir}/{name}_{time_select:.2f}.png")
        plt.close("all")

    def plot_exact_and_predict_in_time(self, param_exact, param_predict, time_snap,
                                       min_value, max_value, name):
        save_dir = f"../logs/{self.model_name}/snapshots/predict_in_time"
        self.create_dir(save_dir)

        plt.figure(figsize=(10, 4))
        v_norm = colors.Normalize(vmin=min_value, vmax=max_value)

        plt.subplot(1, 2, 1)
        plt.imshow(param_exact, cmap="jet", norm=v_norm)
        plt.title(f"Exact {name}(x, y) (t = {time_snap:.2f}s)")
        plt.ylabel("y/D")
        plt.xlabel("x/D")
        plt.colorbar(shrink=0.6)

        plt.subplot(1, 2, 2)
        plt.imshow(param_predict, cmap="jet", norm=v_norm)
        plt.title(f"Learned {name}(x, y) (t = {time_snap:.2f}s)")
        plt.ylabel("y/D")
        plt.xlabel("x/D")
        plt.colorbar(shrink=0.6)
        plt.savefig(f"{save_dir}/{name}_compared_{time_snap:.2f}.png")
        plt.close('all')

    def plot_on_epoch_train(self, params_exact, params_predict, name, selected_time):
        plt.figure(figsize=(8, 6))

        if params_exact is None:
            plt.imshow(params_predict, cmap="jet")
            plt.title(f"Learned {name}(x, y) (t = {selected_time:.2f}s)")
            plt.colorbar(shrink=0.6)
        else:
            plt.subplot(1, 2, 1)
            plt.imshow(params_exact, cmap="jet")
            plt.title(f"Exact {name}(x, y) (t = {selected_time:.2f}s)")
            plt.colorbar(shrink=0.6)

            plt.subplot(1, 2, 2)
            plt.imshow(params_predict, cmap="jet")
            plt.title(f"Learned {name}(x, y) (t = {selected_time:.2f}s)")
            plt.colorbar(shrink=0.6)
        # plt.show()  # Uncomment to display plot when training with spyder IDE

    def make_gif(self, input_dir, start_time, end_time, interval=0.1, name="param", fps=5):
        save_dir = f"../logs/{self.model_name}/animated_gif"
        self.create_dir(save_dir)

        gif_images = []
        n = int((end_time - start_time) / interval) + 1
        time_span = np.linspace(start_time, end_time, n)

        for snap_time in time_span:
            snap_time = round(float(snap_time), 3)
            gif_images.append(
                iio.imread(f"{input_dir}/{name}_{snap_time:.2f}.png")
            )
        iio.mimsave(f"{save_dir}/{name}.gif", gif_images, fps=fps)

    @staticmethod
    def save_checkpoint(model_state, checkpoint_dir):
        ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save(model_state, ckpt)

    @staticmethod
    def callback(epoch, total_loss, loss_u, loss_v, loss_eqn, rel_u, rel_v, lr):
        info = f"Epoch: {epoch + 1:<4d}  " + \
               f"Loss: {total_loss.item():.2e}  " + \
               f"Loss U: {loss_u.item():.2e}  " + \
               f"Loss V: {loss_v.item():.2e}  " + \
               f"Loss Eqns: {loss_eqn.item():.2e}  " + \
               f"Relative Error U: {rel_u.item():.4f}  " + \
               f"Relative Error V: {rel_v.item():.4f}  " + \
               f"LR: {lr:.2e}"
        print(info)

    @staticmethod
    def shuffle_batch(x, y, t, u, v):
        X_total = torch.cat([x, y, t, u, v], dim=1)
        X_total = X_total.clone().numpy()
        np.random.shuffle(X_total)
        return torch.tensor(X_total)

    def save_model(self, model, target_dir):
        model_path = os.path.join(target_dir, f"{self.model_name.lower()}.pth")
        torch.save(model.state_dict(), model_path)

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def relative_error(self, exact, pred):
        return torch.sqrt(torch.mean(torch.square(exact - pred))) / \
               torch.sqrt(torch.mean(torch.square(exact)))

    def plot_loss(self):
        loss_df = pd.read_csv(f"../logs/{self.model_name}/results/losses.csv")
        losses = loss_df.values
        x = np.linspace(1, losses.shape[0], losses.shape[0])[:, None]

        total_loss = losses[:, [0]]
        loss_u = losses[:, [1]]
        loss_v = losses[:, [2]]
        loss_eq = losses[:, [3]]
        rel_u = losses[:, [4]]
        rel_v = losses[:, [5]]

        plt.figure(figsize=(12, 10))
        plt.semilogy(x, total_loss, color="red", label="Total loss")
        plt.semilogy(x, loss_u, color="blue", label=r"$Loss_{u}$")
        plt.semilogy(x, loss_v, color="green", label=r"$Loss_{v}$")
        plt.semilogy(x, loss_eq, color="orange", label=r"$Loss_{PDE}$")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper center", frameon=False, ncol=4)
        plt.savefig(f"../logs/{self.model_name}/results/loss.png", dpi=600)
        plt.show()

        plt.figure(figsize=(12, 10))
        plt.plot(x, rel_u, color="orange", label=r"$L_{2} u(t, x, y)$")
        plt.plot(x, rel_v, color="blue", label=r"$L_{2} v(t, x, y)$")
        plt.ylabel(r"Rel. $L_{2}$ error")
        plt.xlabel("Epoch")
        plt.legend(loc="upper center", frameon=False, ncol=2)
        plt.savefig(f"../logs/{self.model_name}/results/rel_err.png", dpi=600)
        plt.show()

    def predict_drag_lift(self, t_cyl):
        vis = 1.0 / self.Rex
        theta = np.linspace(0., 2 * np.pi, 200).reshape(-1, 1)
        d_theta = theta[1, 0] - theta[0, 0]
        x_cyl = 0.5 * np.cos(theta)
        y_cyl = 0.5 * np.sin(theta)

        Nx = x_cyl.shape[0]
        Nt = t_cyl.shape[0]

        T_star = np.tile(t_cyl, (1, Nx)).T
        X_star = np.tile(x_cyl, (1, Nt))
        Y_star = np.tile(y_cyl, (1, Nt))

        t_star = T_star.reshape(-1, 1)
        x_star = X_star.reshape(-1, 1)
        y_star = Y_star.reshape(-1, 1)

        x_star = torch.tensor(x_star, requires_grad=True, dtype=torch.float32).to(self.device)
        y_star = torch.tensor(y_star, requires_grad=True, dtype=torch.float32).to(self.device)
        t_star = torch.tensor(t_star, requires_grad=True, dtype=torch.float32).to(self.device)

        u, v, p, _, _, _, _ = self.physics_constraints(x_star, y_star, t_star, self.Rex)
        u_x, v_x, u_y, v_y = gradient_velocity_2D(u, v, x_star, y_star)
        del x_star, y_star, t_star

        U_x = to_numpy(u_x).reshape((Nx, Nt))
        U_y = to_numpy(u_y).reshape((Nx, Nt))
        V_x = to_numpy(v_x).reshape((Nx, Nt))
        V_y = to_numpy(v_y).reshape((Nx, Nt))
        P_star = to_numpy(p).reshape((Nx, Nt))
        P_star = P_star - np.mean(P_star, axis=0)

        INT0 = (-P_star[0:-1, :] + 2 * vis * U_x[0:-1, :]) * X_star[0:-1, :] + \
               vis * (U_y[0:-1, :] + V_x[0:-1, :]) * Y_star[0:-1, :]
        INT1 = (-P_star[1:, :] + 2 * vis * U_x[1:, :]) * X_star[1:, :] + vis * (U_y[1:, :] + V_x[1:, :]) * Y_star[1:, :]

        F_D = 0.5 * np.sum(INT0.T + INT1.T, axis=1) * d_theta

        INT0 = (-P_star[0:-1, :] + 2 * vis * V_y[0:-1, :]) * Y_star[0:-1, :] + \
               vis * (U_y[0:-1, :] + V_x[0:-1, :]) * X_star[0:-1, :]
        INT1 = (-P_star[1:, :] + 2 * vis * V_y[1:, :]) * Y_star[1:, :] + vis * (U_y[1:, :] + V_x[1:, :]) * X_star[1:, :]

        F_L = 0.5 * np.sum(INT0.T + INT1.T, axis=1) * d_theta

        return F_D, F_L


if __name__ == "__main__":
    data_path = "../data/cylinder_wake.mat"
    portion = 0.05
    x_train, y_train, t_train, u_train, v_train, minmax_train = read_data_portion(data_path, portion)

    U_s = 0.069
    D = 0.0168
    nu = 1e-6
    Re_D = (U_s * D) / nu

    layers = [3] + 10 * [4 * 5] + [3]
    EPOCHS = 5000
    LAMDA = 1
    LR = 1e-04

    pinn = CylinderResNetHFM(layers_list=layers,
                             activation_name="sine",
                             init_method="xavier_normal",
                             nn_type="resnet",
                             x_data=x_train,
                             y_data=y_train,
                             t_data=t_train,
                             u_data=u_train,
                             v_data=v_train,
                             Re=Re_D,
                             minmax=minmax_train,
                             lamda=LAMDA,
                             epochs=EPOCHS,
                             sample_epoch=100,
                             save_name="Cylinder2D_Wake",
                             learning_rate=LR)

    pinn._train()

    # Testing: Prediction after training network
    model_path = f"../logs/{pinn.model_name}/model/{pinn.model_name.lower()}.pth"
    x_test, y_test, t_test, u_test, v_test, _ = read_data(data_path)
    test_data = torch.cat([x_test, y_test, t_test, u_test, v_test], dim=1).numpy()

    del x_test, y_test, u_test, v_test, x_train, y_train, u_train, v_train, t_train

    pinn.load_state_dict(torch.load(model_path, map_location=pinn.device), strict=False)

    pinn.exact_and_predict_at_selected_time(selected_time=pinn.selected_time, data=test_data)
    pinn.predict_at_selected_time_with_error(data=test_data)

    # Time selection to make animation of predictions
    t_test = t_test.numpy()
    t_start = t_test.min()
    t_end = t_test.max()
    interval = t_test[1] - t_test[0]

    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/predict_in_time",
                  start_time=t_start, end_time=t_end, interval=interval, name="u_compared", fps=10)
    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/predict_in_time",
                  start_time=t_start, end_time=t_end, interval=interval, name="v_compared", fps=10)
    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/p",
                  start_time=t_start, end_time=t_end, interval=interval, name="p", fps=10)
    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/vorticity",
                  start_time=t_start, end_time=t_end, interval=interval, name="vorticity", fps=10)
    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/relative_error",
                  start_time=t_start, end_time=t_end, interval=interval, name="u", fps=5)
    pinn.make_gif(input_dir=f"../logs/{pinn.model_name}/snapshots/relative_error",
                  start_time=t_start, end_time=t_end, interval=interval, name="v", fps=5)

    del t_test, test_data

    # Plot all losses and relative errors
    pinn.plot_loss()
