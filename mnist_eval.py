import os
import datetime
import numpy as np
import torch
from data.load_data import load_mnist, load_fashion_mnist
from models.misvae import MISVAECNN
import argparse
import time
from tqdm import tqdm
import pdb
import re


def evaluate_in_parts(vae, dataloader, L, obj_f, parts=10, convs=False):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0
    if parts > L:
        print(f"parts {parts} > L {L}")
        return
    if convs:
        parts = L
    for x, y in tqdm(dataloader):
        x = x.to(vae.device).float().view((-1, 1, 28, 28))
        if not convs:
            x = x.view((-1, vae.x_dims))
        components = torch.ones(vae.S, device=vae.device)
        with torch.no_grad():
            log_p = []
            log_q = []
            for r in range(parts):
                outputs = vae(x, components, L//parts)
                _, log_p_r, log_q_r = vae.get_log_w(x, *outputs)
                log_p.append(log_p_r)
                log_q.append(log_q_r)
            loss = vae.loss(_, torch.cat(log_p), torch.cat(log_q), L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MISVAE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--estimator', type=str, default='s2s')
    eval_args = parser.parse_args()

    estimator = eval_args.estimator
    device = eval_args.device

    L_final = 1000
    directory_path = 'saved_models/mnist_models/'
    convs = True
    files = [f for f in os.listdir(directory_path) if estimator in f ]

    for file in files:

        args = np.load("saved_models/mnist_models/"+file+"/args.npy", allow_pickle=True).item()
        args.device = device
        vae = MISVAECNN(S=args.S, n_A=args.n_A, lr= args.lr, seed=args.seed, L=args.L, device=args.device, z_dims=args.latent_dims,
                        residual_encoder=args.res_enc, estimator=args.estimator)


        vae.load_state_dict(torch.load(os.path.join("saved_models/mnist_models", file +"/best_model"),
                               map_location=torch.device(device)))

        print("Evaluating by parts")
        _, _, test_dataloader = load_mnist(batch_size_tr=100,
                                                                       batch_size_val=100,
                                                                       batch_size_test=2000)
        avg_elbo = evaluate_in_parts(vae, test_dataloader, L=L_final, obj_f="miselbo", convs=True)

        print("Final NLL: ", avg_elbo, "for file: "+ file)
        np.save(f'{directory_path+file}/test_elbo.npy', avg_elbo)
        print("saved NLL for file "+ file)
