import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import scipy.stats


class DistDataset(Dataset):
    def __init__(self, n_samples=100):
        np.random.seed(0)

        self.d1 = scipy.stats.norm(-2, 1)
        self.d2 = scipy.stats.norm(1.5, 1)
        self.x_min = -2
        self.x_max = 2
        self.y_min = -2
        self.y_max = 2
        self.n_sample_tries = 100

        # generate data
        self.n_samples = n_samples
        # first randomly sample x
        self.x_samp = np.random.uniform(-1, 1, self.n_samples)
        self.y_samp = np.zeros(n_samples)
        # next for each x, sample y
        for i, x in enumerate(self.x_samp):
            self.y_samp[i] = self.sample(x)

    def p(self, x, y):
        a = (x - self.x_min) / (self.x_max - self.x_min)
        return self.d1.pdf((y + .5) * 5 - x) * a + self.d2.pdf((y + .5) * 3 - 2 * x) * (1.0 - a)

    def sample(self, x):
        for i in range(self.n_sample_tries):
            # sample y uniformly
            y_candidate = np.random.uniform(self.y_min, self.y_max)

            # accept or reject sample
            p_y = self.p(x, y_candidate)
            if np.random.rand() <= p_y:
                break

        return y_candidate

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_samp[index:index + 1])
        y = torch.FloatTensor(self.y_samp[index:index + 1])
        alpha = torch.rand(1)
        alpha = alpha * 0.98 + 0.01  # prevent 0 and 1 for numeric stability

        return x, y, alpha


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Tanh()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x, alpha):
        inputs = torch.cat((x,alpha), dim=1)
        hidden1 = self.fc1(inputs)
        hidden2 = self.activation(hidden1)
        hidden2 = self.fc2(hidden2)
        hidden3 = self.activation(hidden2)
        output = self.fc3(hidden3)

        # var = output[:,0:1]
        # cvar_less_var = output[:,1:2]
        # cvar_less_var = torch.sigmoid(cvar_less_var)

        # output = torch.cat((var, cvar_less_var), dim=1)
        # output = self.activation(output)

        return output


class CvarLoss(nn.Module):
    def __init__(self, loss_huber=0.1):
        super(CvarLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # self.l1 = nn.MSELoss()
        self.loss_huber = loss_huber
        self.var_weight = 1.0
        self.cvar_weight = 0.1
        self.mono_weight = 0.001
        self.monotonic_loss_delta = 0.001

    def forward(self, y_pred, gt, alpha, output_alpha_plus):
        var = y_pred[:, 0]
        # cvar_less_var = y_pred[:, 1]
        # cvar = cvar_less_var + var.detach()
        cvar = y_pred[:,1]

        var_loss = self.var_huber_loss(gt, var, alpha)
        cvar_calc = self.cvar_calc(gt, var, alpha)
        cvar_loss = self.l1(cvar, cvar_calc.detach())

        monotonic_loss = 0.0
        var_alpha_plus = output_alpha_plus[:,0]
        monotonic_loss += self.monotonic_loss(var_alpha_plus.detach(), var)
        # cvar_alpha_plus = output_alpha_plus[:,0] + output_alpha_plus[:,1]
        # cvar_alpha_plus = output_alpha_plus[:,1,:,:]
        # monotonic_loss += self.monotonic_loss(cvar_alpha_plus.detach(), cvar)

        return self.var_weight * var_loss + self.cvar_weight * cvar_loss + self.mono_weight * monotonic_loss

        # return self.l1(y_pred[:,0], gt)

    def cvar_calc(self, gt, var, alpha):
        return torch.clamp(gt - var, min=0.0) / (1.0 - torch.clamp(alpha, max=0.99)) + var

    def var_huber_loss(self, gt, var, alpha):
        # compute quantile loss
        err = gt - var
        is_pos_err = torch.lt(var, gt)
        is_neg_err = torch.ge(var, gt)
        is_greater_huber = torch.ge(err, self.loss_huber / alpha)
        is_less_huber = torch.le(err, -self.loss_huber / (1.0 - alpha))

        loss = is_greater_huber * (torch.abs(err) * alpha)
        loss += torch.logical_not(is_greater_huber) * is_pos_err * \
                (0.5 / self.loss_huber * torch.square(alpha * err) + 0.5 * self.loss_huber)
        loss += torch.logical_not(is_less_huber) * is_neg_err * \
                (0.5 / self.loss_huber * torch.square((1.0 - alpha) * err) + 0.5 * self.loss_huber)
        loss += is_less_huber * (torch.abs(err) * (1.0 - alpha))

        return torch.mean(loss)

    def monotonic_loss(self, val_alpha_plus, val):
        diff = torch.clamp(val_alpha_plus - val, max=0.0) / self.monotonic_loss_delta
        smoothed_mae = torch.exp(diff) - diff - 1.0
        return torch.mean(smoothed_mae)

def main():

    # create sample data
    print("creating dataset")
    dataset = DistDataset(n_samples=2000)
    test_dataset = DistDataset(n_samples=100)

    dataloader = DataLoader(dataset,
                           batch_size=2,
                           shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                           batch_size=128,
                           shuffle=True)

    print("building network")
    # create network
    model = Feedforward(2, 256, 2)
    criterion = CvarLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train network!
    print("training")
    epoch = 10
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        for step, (x_input, y_output, alpha) in enumerate(dataloader):
            optimizer.zero_grad()    # Forward pass
            y_pred = model(x_input,alpha)    # Compute Loss
            # print(x_input.transpose(0,1), alpha.transpose(0,1), y_pred.transpose(0,1))
            output_alpha_plus = model(x_input, alpha + 0.01)
            loss = criterion(y_pred, y_output, alpha, output_alpha_plus)
            loss.backward()
            optimizer.step()

            total_loss += loss / float(len(dataloader))

        model.eval()
        test_loss = 0
        for step, (x_input, y_output, alpha) in enumerate(dataloader):
            y_pred = model(x_input,alpha)
            output_alpha_plus = model(x_input, alpha + 0.01)
            test_loss += criterion(y_pred.squeeze(), y_output, alpha, output_alpha_plus) / float(len(dataloader))

        print('Epoch {}: \ttrain: {} \ttest: {}'.format(epoch, total_loss.item(), test_loss.item()))

    # plot!
    plt.subplot(211)
    plt.plot(dataset.x_samp, dataset.y_samp, 'k.', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')

    # evaluate network for plotting
    alpha_test = [0.1, 0.5, 0.9]
    x_input = np.expand_dims(np.linspace(-1, 1, 100), axis=-1)
    x_input = torch.FloatTensor(x_input)
    for alpha_val in alpha_test:
        alpha = alpha_val * torch.ones_like(x_input)
        y_out = model(x_input, alpha)
        var = y_out[:, 0]
        # cvar = y_out[:,0] + y_out[:,1]
        cvar = y_out[:, 1]

        plt.plot(x_input.tolist(), var.tolist(), 'r')
        plt.plot(x_input.tolist(), cvar.tolist(), 'g')

    plt.subplot(212)
    x_lin = [-1, -0.5, 0, 0.5, 1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(x_lin)))

    for i, x in enumerate(x_lin):
        y_lin = np.linspace(-2, 2, 100)
        p_lin = dataset.p(x, y_lin)
        plt.plot(y_lin, p_lin, color=cmap[i])

    leg = [r'$x=$' + str(x) for x in x_lin]
    plt.legend(leg)
    plt.xlabel('y')
    plt.ylabel('p(y)')
    # f.legend(leg, bbox_to_anchor=(0.5, 0.4), loc="center", ncol=4)
    # plt.tight_layout()

    # plt.savefig("huber.pdf", borderaxespad=1, bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    main()
