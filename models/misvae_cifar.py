import numpy as np
import torch
import torch.nn as nn
from models.misvae import MISVAECNN
from models.encoders import EnsembleResnetEncoders
from models.decoders import PixelCNNCIFARDecoder
from torch.distributions import Normal
import torch.nn.functional as F

class MISVAECIFAR(MISVAECNN):
    def __init__(self, S=2, n_A=2, L=1, device='cuda:0', seed=0, n_channels=64,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3, resnet_model='resnet20', n_pixelcnn_layers=4,
                 estimator='s2s'):
        super().__init__(S=S, n_A=n_A, L=L, seed=seed, x_dims=x_dims, z_dims=z_dims, device=device, beta=beta,
                         init_nets=False)
        self.model_name = f"MISVAECIFAR_a_{beta}_seed_{seed}_S_{S}_nA_{n_A}"
        self.obj_f = 'miselbo_beta'

        self.encoder = EnsembleResnetEncoders(latent_dims=z_dims, S=self.S,
                                              resnet_model=resnet_model, device=device).to(device)
        self.decoder = PixelCNNCIFARDecoder(latent_dims=z_dims, n_channels=n_channels, device=device
                                            , n_pixelcnn_layers=n_pixelcnn_layers).to(device)

        self.phi = list(self.encoder.parameters())
        self.theta = self.decoder.parameters()
        self.optim = torch.optim.Adam(params=list(self.phi) + list(self.theta), lr=lr, weight_decay=0)

        self.estimator = estimator

    def forward(self, x, representation, components, L=0):
        if L == 0:
            L = self.L
        if self.estimator == 's2a':
            mu, std = self.encoder(representation, components=torch.ones(self.S, device=self.device))
            mu_out, std_out = mu[:,torch.nonzero(components).flatten()], std[:,torch.nonzero(components).flatten()]
            z = self.sample(mu_out, std_out, L)
        else:
            mu, std = self.encoder(representation, components)
            z = self.sample(mu, std, L)
        reconstruction = self.decoder(x, z)
        return z, mu, std, reconstruction

    def backpropagate(self, x, representations, components):
        z, mu, std, recon = self.forward(x, representations,  components)
        log_w, log_p, log_q = self.get_log_w(x, z, mu, std, recon)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= mu.size(0)  # divide by batch size
        loss.backward()

        # take step
        self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss

    def get_log_w(self, x, z, mu, std, recon):
        # z has dims L, bs, S, z_dims
        L, bs, n_A, S = z.size(0), x.size(0), z.size(-2), mu.size(-2)

        log_px_z = torch.zeros((L, bs, n_A)).to(self.device)
        for l in range(L):
            for s in range(n_A):
                decoder_model_s = DiscMixLogistic(recon[l, :, s])
                log_px_z[l, :, s] = decoder_model_s.log_prob(x)

        # z has dims L, bs, S, z_dims
        log_Q = torch.zeros((L, bs, n_A)).to(self.device)
        # mu has dims 1, bs, S, z_dims
        Q_mixt = Normal(mu, std)

        log_pz = torch.zeros_like(log_Q)
        for s in range(n_A):
            # get z from component s and expand to fit Q_mixt dimensions
            z_s = z[..., s, :].view((L, bs, 1, z.size(-1)))
            # compute likelihood of z_s according to the variational ensemble

            if self.estimator == 's2a':
                log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(S), dim=-1)
            else:
                log_Q_mixture_wrt_z_s = torch.logsumexp(Q_mixt.log_prob(z_s).sum(dim=-1) - np.log(n_A), dim=-1)

            log_Q[..., s] = log_Q_mixture_wrt_z_s
            log_pz[..., s] = self.compute_prior(z_s)

        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        return log_w, log_p, log_Q

    def generate_image(self, n_samples):
        x = torch.zeros((n_samples, 3, 32, 32), device=self.device)
        z = torch.distributions.Normal(0, 1).sample((1, n_samples, 1, self.z_dims)).to(self.device)
        for i in range(32):
            for j in range(32):
                rec = self.decoder(x, z)
                dm = DiscMixLogistic(rec[0, :, 0, :, i, j].view((n_samples, 100, 1, 1)))
                x_samp = dm.sample()
                x[..., i, j] = x_samp[..., 0, 0]
        return x


class DiscMixLogistic:
    # from the NVAE paper's code
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)                    # B, 3, 3 * M, H, W
        self.means = l[:, :, :num_mix, :, :]                                          # B, 3, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   # B, 3, M, H, W
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])              # B, 3, M, H, W
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 3, H , W
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 3, M, H, W
        mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
        mean2 = self.means[:, 1, :, :, :] + \
                self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
        mean3 = self.means[:, 2, :, :, :] + \
                self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
                self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W

        mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
        mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
        mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
        centered = samples - means                          # B, 3, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
        log_probs = torch.logsumexp(log_probs, dim=1)                                 # B, H, W
        return log_probs.sum([-2, -1])                                                # B

    def sample(self, t=1.):
        gumbel = -torch.log(
            # - torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
            - torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).to(self.logit_probs.device)))  # B, M, H, W
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)  # B, M, H, W
        sel = sel.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)  # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)  # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)  # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).to(self.logit_probs.device)  # B, 3, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))  # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)  # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x

    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)  # B, M, H, W
        sel = sel.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)  # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)  # B, 3, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means  # B, 3, H, W
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)  # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).to(indices.device)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot
