import torch
from models.misvae_cifar import MISVAECIFAR, DiscMixLogistic
import os
import matplotlib.pyplot as plt
from data.load_data import load_CIFAR10
import numpy as np
from models.load_pretrained_model import load_resnet


dir_ = '/home/oskar/phd/efficient_mixtures/saved_models/cifar_models/2023-10-01 12:53_MISVAECIFAR_a_1.0_seed_0_S_4_nA_3_lr_0.001_bs_100_warmup_kl_warmup_N_500_epochs_2000_L_1000'
model = MISVAECIFAR(S=4, n_A=3, device='cpu', z_dims=128, n_channels=128, n_pixelcnn_layers=4).to('cpu')
model.load_state_dict(torch.load(os.path.join(dir_, "best_model")))
model.eval()
# resnet = load_resnet('resnet20').cpu()

train_dataloader, val_dataloader, test_dataloader = load_CIFAR10(batch_size_tr=100,
                                                                   batch_size_val=100,
                                                                   batch_size_test=100)

idx = 3
for r, x, y in train_dataloader:
    r = r[idx].float().unsqueeze(0).cpu()
    x_ = x[idx].cpu().unsqueeze(0)
    # x_ *= np.random.binomial(1, 0.9, size=(1, 32, 32))
    x_ = x_.float()
    y = y[idx]

plt.imshow(x_.squeeze(0).T)
plt.show()

with torch.no_grad():
    z, mu, std, reconstruction = model(x_, x_, torch.ones(model.S, device='cpu'))
    # z = torch.distributions.Normal(0, 1).sample((1, 1, 1, model.z_dims))

    for s in range(model.S):
        for _ in range(1):
            x = torch.zeros((1, 3, 32, 32))
            x_in = torch.zeros((1, 3, 32, 32))
            z_s = z[:, :, s].view((1, 1, 1, z.size(-1)))
            for i in range(32):
                for j in range(32):
                    # x[..., :i-1, :j-1] = x_[..., :i-1, :j-1]
                    rec = model.decoder(x_in, z_s)
                    dm = DiscMixLogistic(rec[0, 0, 0, :, i, j].view((1, 100, 1, 1)))
                    x_samp = dm.sample()
                    x[..., i, j] = x_samp[..., 0, 0]
                    x_in[..., i, j] = x_samp[..., 0, 0]
                    # x_in[..., i, j] = x_[..., i, j]


            plt.title(f"Component {s}")
            plt.imshow(x[0].T)
            plt.show()
print()

