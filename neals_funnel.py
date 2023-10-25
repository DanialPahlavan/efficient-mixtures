import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from matplotlib.patches import Ellipse
import time
import matplotlib


class GenerativeModel(nn.Module):
    def __init__(self, device='cuda:0'):
        super(GenerativeModel, self).__init__()
        self.device = device
        self.y_scale_coeff = 9.
        self.x_scale_coeff = 9.

    def log_likelihood(self, z):
        y_N = Normal(0, np.sqrt(self.y_scale_coeff))
        x_N = Normal(0, torch.exp(z[..., 1] / 4))
        return y_N.log_prob(z[..., 1]) + x_N.log_prob(z[..., 0])

    def sample(self, n):
        y_N = Normal(0, np.sqrt(self.y_scale_coeff)).to(self.device)
        y = y_N.sample((n,))
        x_N = Normal(0, np.exp(y / 4)).to(self.device)
        x = x_N.sample((1,))

        return x, y

    def get_contour_plot(self, plot=False):
        xs, ys = torch.meshgrid(
            torch.arange(-10, 10, 0.1),
            torch.arange(-6, 6, 0.1), indexing='ij',
        )
        density = (
                Normal(0.0, (ys / 4.0).exp()).log_prob(xs).exp()
                * Normal(0.0, np.sqrt(self.y_scale_coeff)).log_prob(ys).exp()
        )
        if plot:
            fig, ax = plt.subplots(1, 1)
        plt.contourf(xs, ys, density, levels=[0.0001] + torch.linspace(0.001, 0.1, 10).tolist())
        if plot:
            plt.show()


device = 'cpu'
p = GenerativeModel(device)


def miselbo(z, mu, std):
    log_p = p.log_likelihood(z)
    mix_q = Normal(mu, std)
    log_mix_q_s = torch.zeros((z.size(0), z.size(1)))
    for s in range(z.size(1)):
        z_s = z[:, s].view(-1, 1, 2)
        log_mix_q_s[:, s] = torch.logsumexp(mix_q.log_prob(z_s).sum(-1), dim=-1)
    return -torch.mean(log_p - log_mix_q_s + np.log(z.size(1)))


def miselbo_some_to_all(z, _, __):
    log_p = p.log_likelihood(z)
    components = torch.zeros((S, S), device=device)
    for s in range(S):
        components[s, s] = 1.
    mu, std = components @ W_mu, torch.exp(0.5 * components @ W_log_var)
    mix_q = Normal(mu, std)
    log_mix_q_s = torch.zeros((z.size(0), z.size(1)))
    for s in range(z.size(1)):
        z_s = z[:, s].view(-1, 1, 2)
        log_mix_q_s[:, s] = torch.logsumexp(mix_q.log_prob(z_s).sum(-1), dim=-1)
    return -torch.mean(log_p - log_mix_q_s + np.log(z.size(1)))


# S = 20
estimator = 'some-some'
n_iter = 50000
elbos_N = []
N_list = [10]
S_list = [2]
for S, N in zip(S_list, N_list):
    print("N = ", N)
    t = 0
    elbos = []
    x_axis_epochs = []
    torch.manual_seed(0)
    W_mu = nn.init.kaiming_uniform_(torch.zeros((S, 2), device=device,
                                                requires_grad=True),
                                    nonlinearity='linear'
                                    )
    W_log_var = nn.init.kaiming_uniform_(torch.zeros((S, 2), device=device,
                                                     requires_grad=True),
                                         nonlinearity='linear'
                                         )
    opt = torch.optim.Adam(params=[W_mu] + [W_log_var], lr=0.001)

    for epoch in range(n_iter):
        epoch_start_time = time.time()
        components = torch.zeros((N, S), device=device)
        idx = torch.multinomial(torch.ones(S) / S, N, replacement=False)
        for s in range(N):
            components[s, idx[s]] = 1.
        mu, std = components @ W_mu, torch.exp(0.5 * components @ W_log_var)

        eps = Normal(0, 1).sample((10, N, 2))
        z = mu + std * eps
        if (N == S) or (estimator == 'some-some'):
            loss = miselbo(z, mu, std)
        else:
            loss = miselbo_some_to_all(z, mu, std)
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
                eps = Normal(0, 1).sample((1000 // S, S, 2))
                z = mu + std * eps
                elbo = miselbo(z, mu, std)
                elbos.append(-elbo.numpy())
                x_axis_epochs.append(epoch)
            if epoch % 10000 == 0:
                print("Epoch: ", epoch, "loss", loss.item(), "elbo", elbo)
    figure, axes = plt.subplots()
    matplotlib.rcParams.update({'font.size': 15})
    p.get_contour_plot()
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
    plt.show()
    elbos_N.append(elbos)
    print("\nAverage time/epoch = ", t)
    print()
matplotlib.rcParams.update({'font.size': 15})
for n, elbos in enumerate(elbos_N):
    plt.plot(x_axis_epochs, elbos, label=f'$S={S_list[n]}$')
plt.legend(loc='lower right')
plt.xlabel('Training epochs')
plt.show()

"""
for s in range(S):
    with torch.no_grad():
        v = torch.zeros((1, S))
        v[0, s] = 1
        mu = v @ W_mu
        std = torch.exp(0.5 * v @ W_log_var)
        eps = Normal(0, 1).sample((100, 1, 2))
        z = mu + std * eps
    plt.scatter(z[:, 0, 0].numpy(), z[:, 0, 1].numpy())
plt.show()
"""
