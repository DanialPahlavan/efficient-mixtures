import os
import datetime
import numpy as np
import torch
import torch._dynamo # Keep this import for suppress_errors
from data.load_data import load_mnist, load_fashion_mnist
from models.misvae import MISVAECNN
import argparse
import time
from tqdm import tqdm


def trainer(vae, train_dataloader, val_dataloader, dir_, n_epochs=200,
            verbose=True, L=50, warmup=None, N=100, val_obj_f="miselbo", convs=False):
    if warmup == "kl_warmup":
        vae.beta = 0
    vae.train()
    train_loss_avg = np.zeros(n_epochs)
    eval_loss_avg = []
    best_nll = 1e10
    best_epoch = 0
    training_time = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.
        num_batches = 0

        if warmup == "kl_warmup":
            vae.beta = np.minimum(1 / (N - 1) * epoch, 1.)
        start_time = time.time()

        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            x = x.to(vae.device, non_blocking=True).float().view((-1, 1, 28, 28))
            if not convs:
                x = x.view((-1, vae.x_dims))
            x = torch.bernoulli(x)
            components = torch.zeros(vae.S, device=vae.device)
            idx = torch.multinomial(torch.ones(vae.S) / vae.S, vae.n_A, replacement=False)
            components[idx] = 1.

            loss = vae.backpropagate(x, components)

            epoch_loss += loss
            num_batches += 1
        
        train_loss_avg[epoch] = epoch_loss.item() / num_batches

        epoch_time = time.time() - start_time
        training_time += epoch_time

        test_nll = evaluate(vae, val_dataloader, L=L, obj_f=val_obj_f, convs=convs)
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
            print("Avg. training time", training_time / (epoch + 1))
            if warmup == "kl_warmup":
                print("Beta: ", round(vae.beta, 2))

    return train_loss_avg, eval_loss_avg, training_time, training_time / (epoch + 1)


def evaluate(vae, dataloader, L, obj_f='iwelbo', convs=False):
    if L == 0:
        L = vae.L
    total_elbo = 0.
    total_samples = 0

    for x, y in dataloader:
        x = x.to(vae.device, non_blocking=True).float().view((-1, 1, 28, 28))
        if not convs:
            x = x.view((-1, vae.x_dims))
        with torch.no_grad():
            # components = torch.ones(vae.S, device=vae.device)
            components = torch.zeros(vae.S, device=vae.device)
            idx = torch.multinomial(torch.ones(vae.S) / vae.S, vae.n_A, replacement=False)
            components[idx] = 1.

            outputs = vae(x, components, L)
            log_w, log_p, log_q = vae.get_log_w(x, *outputs)
            loss = vae.loss(log_w, log_p, log_q, L, obj_f=obj_f)
            total_elbo += loss
            total_samples += len(x)
            
    avg_elbo = total_elbo / total_samples
    return avg_elbo.item()


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
        x = x.to(vae.device, non_blocking=True).float().view((-1, 1, 28, 28))
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


def main(args):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    L_final = args.L_final
    n_epochs = args.no_epochs
    batch_size_tr = args.batch_size
    N = 100
    seed = args.seed
    obj_f = 'miselbo'
    device = f"cuda:{args.device}"
    num_workers = args.num_workers

    if args.dataset == 'mnist':
        train_dataloader, val_dataloader, test_dataloader = load_mnist(batch_size_tr=batch_size_tr,
                                                                       batch_size_val=batch_size_tr,
                                                                       batch_size_test=100,
                                                                       num_workers=num_workers)
    elif args.dataset == 'fashion_mnist':
        train_dataloader, val_dataloader, test_dataloader = load_fashion_mnist(batch_size_tr=batch_size_tr,
                                                                               batch_size_val=batch_size_tr,
                                                                               batch_size_test=100,
                                                                               num_workers=num_workers)
    lr = args.lr
    store_path = "saved_models/mnist_models"
    warmup = args.warmup
    vae = MISVAECNN(S=args.S, n_A=args.n_A, lr=lr, seed=seed, L=args.L, device=device, z_dims=args.latent_dims,
                    residual_encoder=args.res_enc, estimator=args.estimator)
    
    # Compile the model for a significant speedup (optional, controlled by --compile_model)
    if args.compile_model:
        torch._dynamo.config.suppress_errors = True # Ensure errors are suppressed if compile is enabled
        vae = torch.compile(vae)

    convs = True

    print("Num. params: ", count_parameters(vae))
    vae.model_name += f"_lr_{lr}_bs_{batch_size_tr}_warmup_{warmup}_nA_{vae.n_A}"
    folder = str(datetime.datetime.now())[0:16] + "_" + vae.model_name + f"_epochs_{n_epochs}_L_{L_final}"
    dir_ = os.path.join(store_path, folder)
    os.makedirs(dir_)
    train_loss, eval_loss, training_time, avg_epoch_time = trainer(
        vae, train_dataloader, val_dataloader, dir_, n_epochs=n_epochs, L=1, warmup=warmup, N=N,
        val_obj_f=obj_f, convs=convs)
    np.save(f'{dir_}/train_loss.npy', train_loss)
    np.save(f'{dir_}/eval_loss.npy', eval_loss)
    np.save(f'{dir_}/args.npy', args)
    np.save(f'{dir_}/training_time.npy', np.array([training_time]))
    np.save(f'{dir_}/avg_epoch_time.npy', np.array([avg_epoch_time]))
    print("\nLoading best model\n")
    vae.load_state_dict(torch.load(os.path.join(dir_, "best_model")))
    print("Evaluating by parts")
    avg_elbo = evaluate_in_parts(vae, test_dataloader, L=args.L_final, obj_f=obj_f, convs=True)
    print("Final ELBO: ", avg_elbo)
    np.save(f'{dir_}/test_elbo.npy', avg_elbo)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MISVAE')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--S', type=int, default=1)
    parser.add_argument('--model', type=str, default='misvaewcnn')
    parser.add_argument('--latent_dims', type=int, default=40)
    parser.add_argument('--warmup', type=str, default='kl_warmup')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--L_final', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--n_A', type=int, default=1)
    parser.add_argument('--res_enc', type=int, default=1)
    parser.add_argument('--estimator', type=str, default='s2s')
    parser.add_argument('--no_epochs', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--compile_model', type=int, default=0, help='Set to 1 to enable torch.compile, 0 to disable.')
    args = parser.parse_args()

    print(args)
    main(args)
            
    #for S, n_A in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (4, 4)]:
    #    args.n_A = n_A
    #    args.S = S
    #    print(args)
    #    main(args)

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
