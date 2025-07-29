import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from models.encoders import EnsembleGatedConv2dEncoders
from models.decoders import SingleLayerPixelCNNDecoder
import pdb

class MISVAECNN(nn.Module):
    def __init__(self, S=2, n_A=2, L=1, device='cuda:0', seed=0,
                 x_dims=784, z_dims=40, beta=1., lr=1e-3, residual_encoder=True, init_nets=True,
                 estimator='s2s'):
        super(MISVAECNN, self).__init__()
        self.model_name = f"{estimator}_seed_{seed}_S_{S}"
        self.obj_f = 'miselbo_beta'
        self.seed = seed
        torch.manual_seed(self.seed)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.device = device
        self.S = S
        self.n_A = n_A  # number of active components
        self.estimator = estimator

        # number of importance samples
        self.L = L
        self.beta = beta

        if init_nets:
            self.encoder = EnsembleGatedConv2dEncoders(n_dims=x_dims, latent_dims=z_dims, h=294, S=S,
                                                       residuals=residual_encoder,
                                                       device=self.device).to(self.device)
            self.decoder = SingleLayerPixelCNNDecoder(x_dims, z_dims, device=device
                                                      ).to(self.device)

            self.phi = list(self.encoder.parameters())
            self.theta = self.decoder.parameters()
            self.optim = torch.optim.Adam(params=list(self.phi) + list(self.theta), lr=lr, weight_decay=0, fused=True)

    def forward(self, x, components, L=0):
        if L == 0:
            L = self.L
        if self.estimator == 's2a':
            mu, std = self.encoder(x, components=torch.ones(self.S, device=self.device))
            mu_out, std_out = mu[:,torch.nonzero(components).flatten()], std[:,torch.nonzero(components).flatten()]
            z = self.sample(mu_out, std_out, L)
        else:
            mu, std = self.encoder(x, components)
            z = self.sample(mu, std, L)
        reconstruction = self.decoder(x, z)
        return z, mu, std, reconstruction

    def sample(self, mu, std, L):
        bs = mu.size(0)
        latent_dims = mu.size(-1)
        n_A = mu.size(-2)
        expanded_shape = (L, bs, n_A, latent_dims)
        eps = torch.randn(expanded_shape).to(self.device)
        mu = mu.unsqueeze(0).expand(expanded_shape)
        std = std.unsqueeze(0).expand(expanded_shape)
        sample = mu + (eps * std)
        return sample.float()

    def loss(self, log_w, log_p=None, log_q=None, L=0, obj_f='elbo'):
        if L == 0:
            L = self.L
        if obj_f == 'elbo':
            elbo = log_w.sum()
            return - elbo
        elif obj_f == 'iwelbo':
            return - torch.sum(torch.logsumexp(log_p - log_q - np.log(L), dim=0))
        elif obj_f == 'miselbo_beta':
            beta_obj = log_w.mean(dim=-1).sum()
            return - beta_obj
        elif obj_f == "miselbo":
            return - torch.sum(torch.mean(torch.logsumexp(log_p - log_q - np.log(L), dim=0), dim=-1))

    def backpropagate(self, x, components, scaler=None):
        z, mu, std, recon = self.forward(x, components)
        log_w, log_p, log_q = self.get_log_w(x, z, mu, std, recon)

        # compute losses
        loss = self.loss(log_w, log_p, log_q, obj_f=self.obj_f)
        loss /= mu.size(0)  # divide by batch size
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
        else:
            loss.backward()
            self.optim.step()

        # reset gradients
        self.optim.zero_grad()
        return loss

    def get_log_w(self, x, z, mu, std, recon):
        # z has dims L, bs, n_A, z_dims
        L, bs, n_A, S = z.size(0), x.size(0), z.size(-2), mu.size(-2)
        x = x.view((1, bs, 1, 784))

        log_px_z = torch.sum(x * torch.log(recon + 1e-8) + (1 - x) * torch.log(1 - recon + 1e-8), dim=-1)

        # z has dims L, bs, n_A, z_dims
        # mu has dims 1, bs, {n_A|S}, z_dims
        Q_mixt = Normal(mu, std)

        # Vectorized computation
        # Reshape z to (L, bs, n_A, 1, z_dims) to broadcast over S components of Q_mixt
        z_expanded = z.unsqueeze(-2)
        
        # log_prob will have shape (L, bs, n_A, S)
        log_prob_z = Q_mixt.log_prob(z_expanded).sum(dim=-1)

        log_denominator = np.log(S) if self.estimator == 's2a' else np.log(n_A)
        
        # log_Q will have shape (L, bs, n_A)
        log_Q = torch.logsumexp(log_prob_z - log_denominator, dim=-1)
        
        # Prior computation, z has shape (L, bs, n_A, z_dims)
        log_pz = self.compute_prior(z)

        log_p = log_px_z + log_pz
        log_w = log_px_z + self.beta * (log_pz - log_Q)
        return log_w, log_p, log_Q

    def compute_prior(self, z):
        # z can be (L, bs, n_A, z_dims) or (L, bs, 1, z_dims)
        if z.dim() == 4 and z.size(2) != 1: # Squeezing only if n_A > 1
             pass
        elif z.dim() == 4 and z.size(2) == 1:
             z = z.squeeze(-2)
        return Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
