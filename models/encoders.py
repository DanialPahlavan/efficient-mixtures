import torch.nn as nn
# import torch.nn.functional as func
import torch
from models.conv_nets import GatedConv2d
from models.load_pretrained_model import load_resnet


class GatedConv2dEncoder(nn.Module):
    def __init__(self, n_dims, S, latent_dims, h=294, activation=nn.ReLU, device='cuda:0'):
        super().__init__()
        self.device = device
        self.activation = activation
        self.h = h
        self.S = S

        self.conv_layer = nn.Sequential(
            GatedConv2d(1, 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )

        self.mu_enc = nn.Linear(in_features=h + self.S, out_features=latent_dims)
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h + self.S, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def forward(self, x, component):
        # component = one-hot encoding
        x = self.conv_layer(x)
        x = x.view((-1, self.h))
        x = torch.cat((x, component), dim=-1)
        mu = self.mu_enc(x)
        std = torch.exp(0.5 * self.log_var_enc(x))
        return mu, std


class GatedConv2dResidualEncoder(nn.Module):
    def __init__(self, n_dims, S, latent_dims, h=294, activation=nn.ReLU, device='cuda:0', conv_layer=True):
        super().__init__()
        self.device = device
        self.activation = activation
        self.h = h
        self.S = S

        if conv_layer:
            self.conv_layer = nn.Sequential(
                GatedConv2d(1, 32, 7, 1, 3),
                GatedConv2d(32, 32, 3, 2, 1),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 3, 2, 1),
                GatedConv2d(64, 6, 3, 1, 1)
            )

        self.mu_0 = nn.Sequential(
            nn.Linear(in_features=h + self.S, out_features=latent_dims),
            nn.ReLU()
        )
        self.mu_1 = nn.Sequential(
            nn.Linear(in_features=h + self.S + latent_dims, out_features=latent_dims),
            nn.ReLU()
        )
        self.mu_enc = nn.Linear(in_features=h + self.S + latent_dims, out_features=latent_dims)

        self.log_var_0 = nn.Sequential(
            nn.Linear(in_features=h + self.S, out_features=latent_dims),
            nn.ReLU()
        )
        self.log_var_1 = nn.Sequential(
            nn.Linear(in_features=h + self.S + latent_dims, out_features=latent_dims),
            nn.ReLU()
        )
        self.log_var_enc = nn.Sequential(
            nn.Linear(in_features=h + self.S + latent_dims, out_features=latent_dims),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

    def _residual_block(self, x_s, layer0, layer1, final_layer):
        x = layer0(x_s)
        x = torch.cat((x_s, x), dim=-1)
        x = layer1(x)
        x = torch.cat((x_s, x), dim=-1)
        return final_layer(x)

    def forward(self, x_s):
        # # x_s = representation before parameterization net + component masking
        # x_mu = self.mu_0(x_s)
        # x_mu = torch.cat((x_s, x_mu), dim=-1)
        # x_mu = self.mu_1(x_mu)
        # x_mu = torch.cat((x_s, x_mu), dim=-1)
        # mu = self.mu_enc(x_mu)
        #
        # x_std = self.log_var_0(x_s)
        # x_std = torch.cat((x_s, x_std), dim=-1)
        # x_std = self.log_var_1(x_std)
        # x_std = torch.cat((x_s, x_std), dim=-1)
        # std = torch.exp(0.5 * self.log_var_enc(x_std))
        # return mu, std

        mu = self._residual_block(x_s, self.mu_0, self.mu_1, self.mu_enc)
        log_var = self._residual_block(x_s, self.log_var_0, self.log_var_1, self.log_var_enc)
        std = torch.exp(0.5 * log_var)
        return mu, std


class EnsembleGatedConv2dEncoders(nn.Module):
    def __init__(self, n_dims, latent_dims, h=294, S=2, residuals=True, activation=nn.ReLU, device='cuda:0',
                 cifar=False):
        super().__init__()
        self.device = device
        self.S = S
        self.latent_dims = latent_dims
        self.residuals = residuals
        if residuals:
            self.encoder = GatedConv2dResidualEncoder(n_dims, S, latent_dims, h=h, activation=activation, device=device)
        else:
            self.encoder = GatedConv2dEncoder(n_dims, S, latent_dims, h=h, activation=activation, device=device)

    def forward(self, x, components):
        # (S, ) = components.shape, components == 1 indicates which component to use
        # This new version is vectorized to be compatible with torch.compile
        bs = x.size(0)
        x = self.encoder.conv_layer(x)
        x = x.view((bs, -1))  # Shape: (bs, h)

        # Prepare for vectorization
        # Repeat x for each of the S components
        x_repeated = x.unsqueeze(1).expand(bs, self.S, self.encoder.h)  # Shape: (bs, S, h)

        # Create one-hot vectors for all S components
        one_hot_components = torch.eye(self.S, device=self.device)  # Shape: (S, S)
        one_hot_components = one_hot_components.unsqueeze(0).expand(bs, self.S, self.S)  # Shape: (bs, S, S)

        # Concatenate x with each component's one-hot vector
        x_s = torch.cat((x_repeated, one_hot_components), dim=-1)  # Shape: (bs, S, h + S)

        # Flatten for processing through linear layers
        x_s_flat = x_s.view(bs * self.S, self.encoder.h + self.S)  # Shape: (bs * S, h + S)

        if self.residuals:
            mu_flat, std_flat = self.encoder(x_s_flat)  # Shape: (bs * S, latent_dims)
        else:
            mu_flat = self.encoder.mu_enc(x_s_flat)
            std_flat = torch.exp(0.5 * self.encoder.log_var_enc(x_s_flat))

        # Reshape back to (bs, S, latent_dims)
        mu = mu_flat.view(bs, self.S, self.latent_dims)
        std = std_flat.view(bs, self.S, self.latent_dims)

        return mu, std


class ResNetEncoder(GatedConv2dResidualEncoder):
    def __init__(self, S, latent_dims, device='cuda:0', resnet_model='resnet1202'):
        super().__init__(n_dims=None, S=S, h=64, latent_dims=latent_dims, device=device, conv_layer=False)
        self.S = S
        self.latent_dims = latent_dims
        self.conv_layer = load_resnet(resnet_model, device)
        # the output dims of the pretrained resnet is 64


class EnsembleResnetEncoders(EnsembleGatedConv2dEncoders):
    def __init__(self, latent_dims, S=2, device='cuda:0', resnet_model='resnet20'):
        super().__init__(n_dims=None, latent_dims=latent_dims, h=64, S=S, device=device)
        self.encoder = ResNetEncoder(S, latent_dims, device, resnet_model)

    def forward(self, x, components):
        # (S, ) = components.shape, components == 1 indicates which component to use
        # This new version is vectorized to be compatible with torch.compile
        bs = x.size(0)
        x = self.encoder.conv_layer(x)  # Shape: (bs, h)

        # Prepare for vectorization
        # Repeat x for each of the S components
        x_repeated = x.unsqueeze(1).expand(bs, self.S, self.encoder.h)  # Shape: (bs, S, h)

        # Create one-hot vectors for all S components
        one_hot_components = torch.eye(self.S, device=self.device)  # Shape: (S, S)
        one_hot_components = one_hot_components.unsqueeze(0).expand(bs, self.S, self.S)  # Shape: (bs, S, S)

        # Concatenate x with each component's one-hot vector
        x_s = torch.cat((x_repeated, one_hot_components), dim=-1)  # Shape: (bs, S, h + S)

        # Flatten for processing through linear layers
        x_s_flat = x_s.view(bs * self.S, self.encoder.h + self.S)  # Shape: (bs * S, h + S)

        if self.residuals:
            mu_flat, std_flat = self.encoder(x_s_flat)  # Shape: (bs * S, latent_dims)
        else:
            mu_flat = self.encoder.mu_enc(x_s_flat)
            std_flat = torch.exp(0.5 * self.encoder.log_var_enc(x_s_flat))

        # Reshape back to (bs, S, latent_dims)
        mu = mu_flat.view(bs, self.S, self.latent_dims)
        std = std_flat.view(bs, self.S, self.latent_dims)

        return mu, std
