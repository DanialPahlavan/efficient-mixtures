import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import matplotlib
import time
from matplotlib.patches import Ellipse
import warnings
import pandas as pd
import pdb
# mute prototype warnings from torch.masked
warnings.filterwarnings(action='ignore', category=UserWarning)


class Prior(nn.Module):
    def __init__(self, Dz, eps=0.001, device='cuda:0', seed=0):
        super(Prior, self).__init__()
        self.device = device
        torch.manual_seed(seed)
        self.Dz = Dz
        self.eps = eps
        self.n1 = Normal(0, eps)
        self.n2 = Normal(0, 1)

    def log_prob(self, z):
        lp = self.n1.log_prob(z[:, :, 0])
        lp += Normal(0, torch.exp(z[:, :, 0] / 4)).log_prob(z[:, :, 1])
        return lp

    def sample(self, n):
        z = torch.zeros((n, self.Dz), device=self.device)
        z[:, 0] = self.n1.sample((n,))
        z[:, 1] = Normal(0, torch.exp(z[:, 0] / 4)).sample((1,))
        return z

    def log_prob_(self, z):
        component_likelihoods = torch.zeros((2, z.shape[0], z.shape[1]), device=self.device)
        component_likelihoods[0] = self.n1.log_prob(z).sum(-1) + np.log(0.5)
        component_likelihoods[1] = self.n2.log_prob(z).sum(-1) + np.log(0.5)
        return torch.logsumexp(component_likelihoods, dim=0)

    def sample_(self, n):
        z = torch.zeros((n, self.Dz), device=self.device)
        for i in range(n):
            u = np.random.uniform(0, 1)
            if u < 0.5:
                z[i] = self.n1.sample((1, self.Dz)).to(self.device)
            else:
                z[i] = self.n2.sample((1, self.Dz)).to(self.device)
        return z


class GenerativeModel(nn.Module):
    def __init__(self, Dz, Dx, eps=0.001, device='cuda:0', seed=0):
        super(GenerativeModel, self).__init__()
        self.device = device
        self.Dz = Dz
        self.Dx = Dx
        torch.manual_seed(seed)
        self.theta = torch.exp(Normal(0, 0.1).sample((Dz, Dx)).to(self.device))
        self.prior = Prior(Dz=Dz, eps=eps, device=device)
        self.decay_rate = 0.1

    def mask(self, y, i, dz):
        m = torch.zeros_like(y, dtype=torch.bool)
        m[:, dz, i] = 1.
        return (y * m).sum(-1).sum(-1).view((y.shape[0], 1))

    def log_likelihood(self, z, x):
        log_prob = self.prior.log_prob(z)
        up_projection = z @ self.theta
        for dz in range(z.shape[1]):
            prob = torch.sigmoid(self.mask(up_projection, 0, dz))
            prob = prob.view((prob.shape[0], prob.shape[1], 1))
            log_prob[:, dz] += torch.sum(torch.log(prob - 1e-4) * x[:, 0] + torch.log((1 - prob + 1e-4)) * (1 - x[:, 0]), -1).squeeze()

            decaying_sequence = x[:, 0].view((-1, 1, 1)) * self.decay_rate
            for i in range(1, self.Dx):

                prob = torch.sigmoid(self.mask(up_projection, i, dz) + decaying_sequence)
                prob = prob.view((prob.shape[0], prob.shape[1]))
                log_prob[:, dz] += torch.sum(torch.log(prob - 1e-4) * x[:, i].view((-1, 1)) +
                                      torch.log((1 - prob + 1e-4)) * (1 - x[:, i].view((-1, 1))),
                                      dim=0)

                decaying_sequence += x[:, i].view((-1, 1, 1))  # torch.abs(self.mask(up_projection, i, dz))
                decaying_sequence *= self.decay_rate
        return log_prob

    def sample(self, n):
        z = self.prior.sample(n)
        up_projection = z @ self.theta
        x = torch.zeros((n, self.Dx), device=self.device)
        x[:, 0] = torch.bernoulli(torch.sigmoid(up_projection[:, 0]))
        decaying_sequence = x[:, 0] * self.decay_rate  # torch.abs(up_projection[:, 0]) * self.decay_rate
        for i in range(1, self.Dx):
            # temperature = x[:, :i].sum(-1) + 1

            prob = torch.sigmoid(up_projection[:, i] + decaying_sequence)
            x[:, i] = torch.bernoulli(prob)

            decaying_sequence += x[:, i]
            decaying_sequence *= self.decay_rate
        return x, z

    def get_contour_plot(self, x, plot=False):
        xs, ys = torch.meshgrid(
            torch.arange(-10, 10, 0.1),
            torch.arange(-6, 6, 0.1), indexing='ij',
        )
        z = torch.zeros()
        density = (
                self.log_likelihood(z, x)
        )
        if plot:
            fig, ax = plt.subplots(1, 1)
        plt.contourf(xs, ys, density, levels=[0.0001] + torch.linspace(0.001, 0.1, 10).tolist())
        if plot:
            plt.show()


def miselbo(z, x, mu, std, p):
    log_p = p.log_likelihood(z, x).to(device)
    mix_q = Normal(mu, std)
    log_mix_q_s = torch.zeros((z.size(0), z.size(1)), device=device)
    for s in range(z.size(1)):
        z_s = z[:, s].view(-1, 1, 2)
        log_mix_q_s[:, s] = torch.logsumexp(mix_q.log_prob(z_s).sum(-1), dim=-1)
    return -torch.mean(log_p - log_mix_q_s + np.log(z.size(1)))


def miselbo_some_to_all(z, x, _, __, p):
    log_p = p.log_likelihood(z, x).to(device)
    components = torch.zeros((S, S), device=device)
    for s in range(S):
        components[s, s] = 1.
    mu, std = components @ W_mu, torch.exp(0.5 * components @ W_log_var)
    mix_q = Normal(mu, std)
    log_mix_q_s = torch.zeros((z.size(0), z.size(1)), device=device)
    for s in range(z.size(1)):
        z_s = z[:, s].view(-1, 1, 2)
        log_mix_q_s[:, s] = torch.logsumexp(mix_q.log_prob(z_s).sum(-1), dim=-1)
    return -torch.mean(log_p - log_mix_q_s + np.log(S))


device = 'cpu'
p = GenerativeModel(2, 20, eps=3, device=device)
x, _ = p.sample(5)
_, z = p.sample(10000)
z = z.view((z.shape[0], 1, -1))
ll = p.log_likelihood(z, x)
z_vis = z.to('cpu').squeeze().numpy()
ll = ll.to('cpu').squeeze().numpy()
sc = plt.scatter(z_vis[:, 0], z_vis[:, 1], c=ll)
# plt.colorbar(sc)
"""
matplotlib.rcParams.update({'font.size': 15})
plt.xlabel('$z_1$', fontsize=20)
plt.ylabel('$z_2$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()

# Create a mask that is True wherever ll is NOT NaN
not_nan_mask = ~np.isnan(ll)

# Use the mask to filter both ll and z_vis to exclude rows with NaNs in ll
filtered_ll = ll[not_nan_mask]
filtered_z_vis = z_vis[not_nan_mask]

# Verify the shapes
print(filtered_ll.shape)
print(filtered_z_vis.shape)
#plt.show()

# Convert to a DataFrame
df = pd.DataFrame(np.concatenate((filtered_z_vis, filtered_ll.reshape(-1, 1)), axis=1), columns=['z1', 'z2','ll'])

# Select every tenth row
df_reduced = df.iloc[::3, :]

# Save to CSV
df_reduced.to_csv('z_vis_big.csv', index=False)
"""


estimators = ['s2a']
n_iter = 50000
elbos_N = []
#N_list = [1, 5, 5, 2,1]
#S_list = [5, 5, 20, 5,20]

N_list = [2]
S_list = [5]

t_list = []
for e, (S, N) in enumerate(zip(S_list, N_list)):
    estimator = estimators[e]
    best_elbo = -1000000.
    best_epoch = 0
    print("N = ", N)
    t = 0
    elbos = []
    x_axis_epochs = []
    torch.manual_seed(1)
    W_mu = nn.init.kaiming_uniform_(torch.zeros((S, 2), device=device,
                                                requires_grad=True),
                                    nonlinearity='linear'
                                    )
    W_log_var = nn.init.kaiming_uniform_(torch.zeros((S, 2), device=device,
                                                     requires_grad=True),
                                         nonlinearity='linear'
                                         )
    opt = torch.optim.Adam(params=[W_mu] + [W_log_var], lr=0.001)

    for epoch in range(n_iter + 1):
        epoch_start_time = time.time()
        components = torch.zeros((N, S), device=device)
        idx = torch.multinomial(torch.ones(S) / S, N, replacement=False)
        for s in range(N):
            components[s, idx[s]] = 1.
        mu, std = components @ W_mu, torch.exp(0.5 * components @ W_log_var)

        eps = Normal(0, 1).sample((10, N, 2)).to(device)
        z = mu + std * eps
        if (N == S) or (estimator == 's2s'):
            loss = miselbo(z, x, mu, std, p)
        else:
            loss = miselbo_some_to_all(z, x, mu, std, p)
        loss.backward()
        opt.step()
        opt.zero_grad()
        t += (time.time() - epoch_start_time) / n_iter
        if epoch % 100 == 0:
            with torch.no_grad():
                components = torch.zeros((S, S), device=device)
                for s in range(S):
                    components[s, s] = 1.
                mu, std = components @ W_mu, torch.exp(0.5 * components @ W_log_var)
                eps = Normal(0, 1).sample((1000 // S, S, 2)).to(device)
                z = mu + std * eps
                elbo = miselbo(z, x, mu, std, p)
                elbos.append(-elbo.cpu().numpy())
                x_axis_epochs.append(epoch)
                if best_elbo < -elbo:
                    best_elbo = -elbo
                    best_epoch = epoch
            if epoch % 1000 == 0:
                # print("Best elbo: ", best_elbo.item(), "Best epoch: ", best_epoch)
                print("Epoch: ", epoch, "loss", loss.item(), "elbo", -elbo.item())
    figure, axes = plt.subplots()
    matplotlib.rcParams.update({'font.size': 15})
    # p.get_contour_plot()
    sc = plt.scatter(z_vis[:, 0], z_vis[:, 1], c=ll)
    for s in range(S):
        with torch.no_grad():
            v = torch.zeros((1, S))
            v[0, s] = 1
            mu = v @ W_mu
            std = torch.exp(0.5 * v @ W_log_var)
        mu_s = mu.detach().cpu().numpy()
        std_s = std.detach().cpu().numpy()
        circle = Ellipse((mu_s[0, 0], mu_s[0, 1]), 2 * std_s[0, 0], 2 * std_s[0, 1],
                         fill=False, color='red')

        # axes.set_aspect(1)
        axes.add_artist(circle)
    # plt.legend()
    matplotlib.rcParams.update({'font.size': 15})
    plt.title(f"$N$={N}, $S$={S}")
    plt.xlabel('$z_1$', fontsize=20)
    plt.ylabel('$z_2$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    #plt.show()
    elbos_N.append(elbos)
    print("\nAverage time/epoch = ", t)
    t_list.append(t)
    print("Best elbo, epoch and time", (best_elbo, best_epoch, t * best_epoch))
    print()



plt.figure()
matplotlib.rcParams.update({'font.size': 15})
x_axis_epochs = np.array(x_axis_epochs)
colors = ['blue', 'orange', 'green', 'black', 'red']



for n, elbos in enumerate(elbos_N):
    plt.plot(t_list[n] * x_axis_epochs, elbos, label=f'$S={N_list[n]}, A={S_list[n]}$ ({estimators[n]})', color=colors[n], alpha=0.5)
    i = np.argmax(elbos)
    plt.plot(t_list[n] * x_axis_epochs[i], elbos[i], color=colors[n], marker='*')
plt.legend(loc='lower right')
plt.ylabel('MISELBO')
plt.xlabel('Training time')
plt.tight_layout()
plt.show()


import csv

combinations = set(zip(N_list, S_list, estimators))

# For each combination, create a separate CSV file
for combo in combinations:
    N, S, estimator = combo
    filename = f"data_S{N}_A{S}_{estimator}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'ELBO'])
        for n, elbos in enumerate(elbos_N):
            if N_list[n] == N and S_list[n] == S and estimators[n] == estimator:
                for i, elbo in enumerate(elbos):
                    time = t_list[n] * x_axis_epochs[i]
                    writer.writerow([time, elbo])
