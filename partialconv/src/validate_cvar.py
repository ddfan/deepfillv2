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
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        hidden1 = self.fc1(x)
        hidden2 = self.activation(hidden1)
        hidden2 = self.fc2(hidden2)
        hidden3 = self.activation(hidden2)
        output = self.fc3(hidden3)

        return output


class CvarLoss(nn.Module):
    def __init__(self, loss_huber=0.1):
        super(CvarLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.loss_huber = loss_huber
        self.var_weight = 10.0
        self.cvar_weight = 1.0

    def forward(self, y_pred, gt, alpha):
        var = y_pred[:, 0]
        cvar_less_var = y_pred[:, 1]
        cvar = cvar_less_var + var.detach()

        var_loss = self.var_huber_loss(gt, var, alpha)
        cvar_calc = self.cvar_calc(gt, var, alpha)
        cvar_loss = self.l1(cvar, cvar_calc.detach())

        return self.var_weight * var_loss + self.cvar_weight * cvar_loss

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


def main():

    # create sample data
    print("creating dataset")
    dataset = DistDataset(n_samples=1000)
    test_dataset = DistDataset(n_samples=100)

    dataloader = DataLoader(dataset,
                           batch_size=128,
                           shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                           batch_size=128,
                           shuffle=True)

    print("building network")
    # create network
    model = Feedforward(1, 64, 2)
    criterion = CvarLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    # train network!
    print("training")
    epoch = 1000
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        for step, (x_input, y_output, alpha) in enumerate(dataloader):
            optimizer.zero_grad()    # Forward pass
            y_pred = model(x_input)    # Compute Loss
            loss = criterion(y_pred, y_output, alpha)

            loss.backward()
            optimizer.step()

            total_loss += loss / float(len(dataloader))

        model.eval()
        test_loss = 0
        for step, (x_input, y_output, alpha) in enumerate(dataloader):
            y_pred = model(x_input)
            test_loss += criterion(y_pred.squeeze(), y_output, alpha) / float(len(dataloader))

        print('Epoch {}: \ttrain: {} \ttest: {}'.format(epoch, total_loss.item(), test_loss.item()))

    # evaluate network for plotting

    # plot!
    plt.subplot(211)
    plt.plot(dataset.x_samp, dataset.y_samp, 'k.', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')

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
