import torch
import torch.nn as nn
from models.fc_nets import GatedDense
from models.conv_nets import MaskedConv2d, Conv2d, MaskedGatedConv2d
from collections import OrderedDict


class TwoLayerDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0', gated=True):
            super().__init__()
            self.device = device
            self.activation = activation
            if not gated:
                self.fc_layers = nn.Sequential(
                    nn.Linear(in_features=latent_dims, out_features=h_dim_2),
                    self.activation(),
                    nn.Linear(in_features=h_dim_2, out_features=h_dim_1),
                    self.activation(),

                )
            else:
                self.fc_layers = nn.Sequential(
                    GatedDense(latent_dims, h_dim_2),
                    GatedDense(h_dim_2, h_dim_1)
                )
            self.bernoulli_dec = nn.Linear(in_features=h_dim_1, out_features=n_dims)

    def forward(self, z):
        x = self.fc_layers(z)
        return torch.sigmoid(self.bernoulli_dec(x))


class OneLayerDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim=20, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device
            self.activation = activation
            self.fc_layer = nn.Sequential(
                            nn.Linear(in_features=latent_dims, out_features=h_dim),
                            self.activation(),
                        )
            self.bernoulli_dec = nn.Linear(in_features=h_dim, out_features=n_dims)

    def forward(self, z):
        x = self.fc_layer(z)
        return torch.sigmoid(self.bernoulli_dec(x))


class PixelCNNDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, h_dim_1=300, h_dim_2=300, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device
            self.activation = activation

            # p(z1|z2)
            self.fc_layers_lower = nn.Sequential(
                GatedDense(latent_dims, h_dim_2),
                GatedDense(h_dim_2, h_dim_1)
            )

            self.mu_z1 = nn.Linear(in_features=h_dim_1, out_features=latent_dims)
            self.log_var_z1 = nn.Sequential(
                nn.Linear(in_features=h_dim_1, out_features=latent_dims),
                nn.Hardtanh(min_val=-6., max_val=2.)
            )

            self.p_x_layers_z1 = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )
            self.p_x_layers_z2 = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )

            # PixelCNN
            act = nn.ReLU(True)
            self.pixelcnn = nn.Sequential(
                MaskedConv2d('A', 1 + 2 * 1, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
            )

            self.bernoulli_dec = Conv2d(64, 1, 1, 1, 0)

    def forward(self, x_in, z1, z2):
        z2_ = self.fc_layers_lower(z2)
        mu = self.mu_z1(z2_)
        std = torch.exp(0.5 * self.log_var_z1(z2_))

        z2 = self.p_x_layers_z2(z2)
        z1 = self.p_x_layers_z1(z1)

        # L, bs, S, 1, 28, 28
        x_out = torch.zeros((z1.size(0), z1.size(1), z1.size(2), 1, 28, 28), device=self.device)
        z1 = z1.view((z1.size(1), z1.size(2), 1, 28, 28))
        z2 = z2.view((z2.size(1), z2.size(2), 1, 28, 28))

        for s in range(x_out.size(2)):
            x = torch.cat((x_in, z1[:, s], z2[:, s]), dim=-3)
            x = self.pixelcnn(x)
            x_out[:, :, s] = torch.sigmoid(self.bernoulli_dec(x))
        x_out = x_out.view((x_out.size(0), x_out.size(1), x_out.size(2), 784))
        return x_out, mu, std


class SingleLayerPixelCNNDecoder(nn.Module):
    def __init__(self, n_dims, latent_dims, activation=nn.ReLU, device='cuda:0'):
            super().__init__()
            self.device = device

            self.p_x_layers = nn.Sequential(
                GatedDense(latent_dims, n_dims)
            )

            # PixelCNN
            act = activation(True)
            self.pixelcnn = nn.Sequential(
                MaskedConv2d('A', 1 + 1, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
                MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
            )

            self.bernoulli_dec = Conv2d(64, 1, 1, 1, 0)

    def forward(self, x_in, z):
        z = self.p_x_layers(z)

        # L, bs, S, 1, 28, 28
        x_out = torch.zeros((z.size(0), z.size(1), z.size(2), 1, 28, 28), device=self.device)
        z = z.view((z.size(1), z.size(2), 1, 28, 28))

        for s in range(x_out.size(2)):
            x = torch.cat((x_in, z[:, s]), dim=-3)
            x = self.pixelcnn(x)
            x_out[:, :, s] = torch.sigmoid(self.bernoulli_dec(x))
        x_out = x_out.view((x_out.size(0), x_out.size(1), x_out.size(2), 784))
        return x_out


class PixelCNNCIFARDecoder(nn.Module):
    def __init__(self, latent_dims, device='cuda:0', n_channels=64, n_pixelcnn_layers=4):
        super(PixelCNNCIFARDecoder, self).__init__()
        self.device = device

        self.linear = nn.init.kaiming_uniform_(torch.zeros((latent_dims, n_channels, 4, 4), device=device,
                                               requires_grad=True),
                                               nonlinearity='linear'
        )

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels, n_channels, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels, n_channels, 3, 2, 1, 1),
            nn.ReLU(),
        )

        act = nn.ReLU()

        self.pixelcnn = nn.Sequential(OrderedDict([
            ('MCNN_A',
             nn.Sequential(MaskedGatedConv2d('A', 3, n_channels, 3, 1, 1, bias=False)))] +
            [(f'MCNN_B_{i}', nn.Sequential(MaskedGatedConv2d('B', n_channels, n_channels, 3, 1, 1, bias=False)))
             for i in range(n_pixelcnn_layers - 1)]
        )
        )
        """
        self.pixelcnn = nn.Sequential(MaskedConv2d('A', n_channels + 3, n_channels, 3, 1, 1,
                                                                        bias=False),
                                                           nn.BatchNorm2d(n_channels),
                                                           act,
                                                      MaskedGatedConv2d('B', n_channels, n_channels, 3, 1, 1,
                                                                        bias=False),
                                                      nn.BatchNorm2d(n_channels),
                                                      act,
                                                    MaskedGatedConv2d('B', n_channels, n_channels, 3, 1, 1,
                                                                      bias=False),
                                                    nn.BatchNorm2d(n_channels),
                                                    act,
                                                    MaskedGatedConv2d('B', n_channels, n_channels, 3, 1, 1,
                                                                      bias=False),
                                                    nn.BatchNorm2d(n_channels),
                                                    act
                                                    )
        """
        # number of logistic mixtures = 10 with 10 params each
        self.cnn = nn.Conv2d(n_channels, 100, 1, 1)

    def forward(self, x, z):
        # expects z to have shape (L, B, S, latent_dims), where B is batch size
        z = torch.einsum('zchw, lbsz -> lbschw', self.linear, z)
        x_out = torch.zeros((z.size(0), z.size(1), z.size(2), 100, 32, 32), device=self.device)

        for l in range(z.size(0)):
            for s in range(z.size(2)):
                z_ls = z[l, :, s]
                z_ls = self.conv_transpose(z_ls)
                # y = torch.cat([x, z_ls], dim=-3)
                y = x
                # z_ls = self.pixelcnn(z_ls)
                for i, m in enumerate(self.pixelcnn):
                    y = m[0](y, z_ls)
                    # y = torch.cat([y, z_ls], dim=-3)
                x_out[l, :, s] = self.cnn(y)

        return x_out

