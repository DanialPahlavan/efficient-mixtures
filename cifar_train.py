import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from data.load_data import load_CIFAR10
from models.misvae_cifar import MISVAECIFAR
import argparse
import time
from tqdm import tqdm
from models.load_pretrained_model import load_resnet


def trainer(vae, train_dataloader, val_dataloader, dir_, n_epochs=200,
            verbose=True, L=1, warmup=None, N=100, val_obj_f="miselbo"):
    resnet = load_resnet('resnet20').to(vae.device)
    if warmup == "kl_warmup":
        vae.beta = 0
    vae.train()
    train_loss_avg = np.zeros(n_epochs)
    eval_loss_avg = []
    best_nll = 1e10
    best_epoch = 0
    training_time = 0

    for epoch in range(n_epochs):
        num_batches = 0

        if warmup == "kl_warmup":
            vae.beta = np.minimum(1 / (N - 1) * epoch, 1.)
        start_time = time.time()

        for r, x, y in train_dataloader:
            # r = r.to(vae.device).float()
            x = x.to(vae.device).float()

            components = torch.zeros(vae.S, device=vae.device)
            idx = torch.multinomial(torch.ones(vae.S) / vae.S, vae.n_A, replacement=False)
            components[idx] = 1.

            loss = vae.backpropagate(x, x, components)

            train_loss_avg[epoch] += loss.item()
            num_batches += 1

        epoch_time = time.time() - start_time
        training_time += epoch_time

        test_nll = evaluate(vae, val_dataloader, L=L, obj_f=val_obj_f, resnet=resnet)
        train_loss_avg[epoch] /= num_batches
        eval_loss_avg.append(test_nll)
        if test_nll < best_nll:
            path = os.path.join(dir_, "best_model")
            torch.save(vae.state_dict(), path)
            best_nll = test_nll
            best_epoch = epoch
        # elif (epoch - best_epoch) >= 100:
        #     return train_loss_avg, eval_loss_avg, training_time, training_time / (epoch + 1)

        if verbose and epoch % 10 == 0:
            print("Epoch: ", epoch)
            print(f"Test NLL: ", test_nll, f" ({round(best_nll, 2)}; {best_epoch})")
            print(f"Test BPD: ", test_nll / (32 ** 2 * 3) / np.log(2),
                  f" ({round(best_nll / (32 ** 2 * 3) / np.log(2), 2)}; {best_epoch})")
            print("Avg. training time", training_time / (epoch + 1))
            if epoch % 100:
                with torch.no_grad():
                    vae.eval()
                    gens = vae.generate_image(5)
                    for v, img in enumerate(gens):
                        plt.imsave(dir_ + f"/{epoch}{v}.png", torch.transpose(img.T, 0, 1).detach().cpu().numpy())
                    vae.train()
            if warmup == "kl_warmup":
                print("Beta: ", round(vae.beta, 2))

    return train_loss_avg, eval_loss_avg, training_time, training_time / (epoch + 1)


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


def main(args):
    L_final = args.L_final
    n_epochs = 2000
    batch_size_tr = args.batch_size
    N = 500
    seed = args.seed
    obj_f = 'miselbo'
    device = f"cuda:{args.device}"

    train_dataloader, val_dataloader, test_dataloader = load_CIFAR10(batch_size_tr=batch_size_tr,
                                                                   batch_size_val=batch_size_tr,
                                                                   batch_size_test=100)

    lr = args.lr
    store_path = "saved_models/cifar_models"
    warmup = args.warmup
    vae = MISVAECIFAR(S=args.S, n_A=args.n_A, lr=lr, seed=seed, L=args.L, device=device,
                      z_dims=args.latent_dims, n_channels=args.n_channels, n_pixelcnn_layers=args.n_pixelcnn_layers)
    print("Num. params: ", count_parameters(vae))

    vae.model_name += f"_lr_{lr}_bs_{batch_size_tr}_warmup_{warmup}_N_{N}"
    folder = str(datetime.datetime.now())[0:16] + "_" + vae.model_name + f"_epochs_{n_epochs}_L_{L_final}"
    dir_ = os.path.join(store_path, folder)
    os.makedirs(dir_)

    train_loss, eval_loss, training_time, avg_epoch_time = trainer(
        vae, train_dataloader, val_dataloader, dir_, n_epochs=n_epochs, L=1, warmup=warmup, N=N,
        val_obj_f=obj_f)

    np.save(f'{dir_}/train_loss.npy', train_loss)
    np.save(f'{dir_}/eval_loss.npy', eval_loss)
    np.save(f'{dir_}/args.npy', args)
    np.save(f'{dir_}/training_time.npy', np.array([training_time]))
    np.save(f'{dir_}/avg_epoch_time.npy', np.array([avg_epoch_time]))

    print("\nLoading best model\n")
    vae.load_state_dict(torch.load(os.path.join(dir_, "best_model")))
    print("Evaluating by parts")
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=args.L_final, obj_f=obj_f, convs=True)

    print("Final NLL: ", avg_elbo)
    np.save(f'{dir_}/test_elbo.npy', avg_elbo)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MISVAE')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--S', type=int, default=1)
    parser.add_argument('--model', type=str, default='misvaewcnn')
    parser.add_argument('--latent_dims', type=int, default=256)
    parser.add_argument('--warmup', type=str, default='kl_warmup')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--L_final', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--n_A', type=int, default=1)
    parser.add_argument('--res_enc', type=int, default=1)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--n_pixelcnn_layers', type=int, default=4)
    args = parser.parse_args()

    for S, n_A in [(4, 3)]:
        args.n_A = n_A
        args.S = S
        print(args)
        main(args)

    """
    vae = MISVAECNN(S=S)
    convs = True

    print('Loading best model')
    vae.load_state_dict(torch.load(os.path.join("/home/oskar/phd/efficient_mixtures/saved_models/mnist_models/"
                                                "2023-08-22 16:06_MISVAEwCNN_a_1.0_seed_0_S_1_lr_0.0005_bs_100_warmup_kl_warmup_N_100_epochs_4000_L_1000",
                                                "best_model")))

    print("Evaluating by parts")
    train_dataloader, val_dataloader, test_dataloader = load_mnist(batch_size_tr=100,
                                                                   batch_size_val=100,
                                                                   batch_size_test=2000)
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=1000, obj_f="miselbo", convs=True)
    # avg_elbo = evaluate(vae, test_dataloader, L=5000, obj_f="miselbo")
    print("Final NLL: ", avg_elbo)
    """












