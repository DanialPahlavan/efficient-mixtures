import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from data.load_data import load_CIFAR10, load_CIFAR10_small
from models.misvae_cifar import MISVAECIFAR
import argparse
import time
from tqdm import tqdm
from models.load_pretrained_model import load_resnet


def evaluate(vae, dataloader, L, resnet, obj_f='miselbo'):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0

    for r, x, y in dataloader:
        # r = r.to(vae.device).float()
        x = x.to(vae.device).float()
        with torch.no_grad():
            components = torch.ones(vae.S, device=vae.device)
            # components = torch.zeros(vae.S, device=vae.device)
            # idx = torch.multinomial(torch.ones(vae.S) / vae.S, vae.n_A, replacement=False)
            # components[idx] = 1.

            outputs = vae(x, x, components, L)
            log_w, log_p, log_q = vae.get_log_w(x, *outputs)
            loss = vae.loss(log_w, log_p, log_q, L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def evaluate_in_parts(vae, dataloader, L, obj_f, parts=10, convs=True):
    if L == 0:
        L = vae.L
    elbo = 0
    num_batches = 0
    if parts > L:
        print(f"parts {parts} > L {L}")
        return
    if convs:
        parts = L
    for r, x, y in tqdm(dataloader):
        # r = r.to(vae.device).float()
        x = x.to(vae.device).float()
        components = torch.ones(vae.S, device=vae.device)
        with torch.no_grad():
            log_p = []
            log_q = []
            for part in range(parts):
                outputs = vae(x, x, components, L//parts)
                _, log_p_part, log_q_part = vae.get_log_w(x, *outputs)
                log_p.append(log_p_part)
                log_q.append(log_q_part)
            loss = vae.loss(_, torch.cat(log_p), torch.cat(log_q), L, obj_f=obj_f)
            elbo += loss.item()
            num_batches += len(x)
    avg_elbo = elbo / num_batches
    return avg_elbo


def main(eval_args):
    L_final = 1000
    directory_path = 'saved_models/cifar_models/'
    files = [f for f in os.listdir(directory_path) if "epochs" in f ]


    N = 500
    obj_f = 'miselbo'
    device = f"cuda:{eval_args.device}" if torch.cuda.is_available() else 'cpu'

    _, _, test_dataloader = load_CIFAR10(batch_size_tr=100, batch_size_val=100,
                                        batch_size_test=100)

    for file in files:
        args = np.load(directory_path+file+"/args.npy", allow_pickle=True).item()
        args.device = device

        vae = MISVAECIFAR(S=args.S, n_A=args.n_A, lr=args.lr, seed=args.seed, L=args.L, device=args.device,
                          z_dims=args.latent_dims, n_channels=args.n_channels,
                           n_pixelcnn_layers=args.n_pixelcnn_layers, estimator=args.estimator)

        vae.load_state_dict(torch.load(os.path.join(directory_path, file +"/best_model"),
                               map_location=torch.device(device)))

        print("Evaluating by parts")
        avg_elbo = evaluate_in_parts(vae, test_dataloader, L=L_final, obj_f="miselbo", convs=True)


        print("Final NLL: ", avg_elbo, "for file: "+ file)
        np.save(f'{directory_path+file}/test_elbo.npy', avg_elbo)
        print("saved NLL for file "+ file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MISVAE')
    parser.add_argument('--device', type=int, default=0)
    eval_args = parser.parse_args()
    main(eval_args)
