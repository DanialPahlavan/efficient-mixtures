# Efficient Mixture Learning in Black-Box Variational Inference
Code for reproducing results for [Efficient Mixture Learning in Black-Box Variational Inference](https://arxiv.org/pdf/2406.07083).

## Abstract
Mixture variational distributions in black box variational inference (BBVI) have demonstrated impressive results in challenging density estimation tasks. However, currently scaling the number of mixture components can lead to a linear increase in the number of learnable parameters and a quadratic increase in inference time due to the evaluation of the evidence lower bound (ELBO). Our two key contributions address these limitations. First, we introduce the novel Multiple Importance Sampling Variational Autoencoder (*MISVAE*), which amortizes the mapping from input to mixture-parameter space using one-hot encodings. Fortunately, with *MISVAE*, each additional mixture component incurs a negligible increase in network parameters. Second, we construct two new estimators of the ELBO for mixtures in BBVI, enabling a tremendous reduction in inference time with marginal or even improved impact on performance. Collectively, our contributions enable scalability to hundreds of mixture components and provide superior estimation performance in shorter time, with fewer network parameters compared to previous Mixture VAEs. Experimenting with *MISVAE*, we achieve astonishing, SOTA results on MNIST. Furthermore, we empirically validate our estimators in other BBVI settings, including Bayesian phylogenetic inference, where we improve inference times for the SOTA mixture model on eight data sets. 

## How to train on MNIST
**Note** that in the codebase, the parameter `--S` corresponds to 'A' in the paper terminology, and `--n_A` corresponds to 'S' in the paper. The `--estimator` parameter can be set to either 's2a' or 's2s', depending on the desired estimator. To train on MNIST with $A=20$ and $S=2$ using the 's2a' estimator, for example, one would use the command:

```
python -m mnist_train --L_final 1000 --S 20 --estimator s2a --n_A 2
```

### Command-Line Arguments

The following arguments can be used to configure the MNIST training script:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--batch_size` | The batch size for training. | 100 |
| `--S` | Corresponds to 'A' in the paper (number of mixture components). | 1 |
| `--model` | The model to use. | 'misvaewcnn' |
| `--latent_dims` | The number of latent dimensions. | 40 |
| `--warmup` | The warmup strategy for the KL divergence. | 'kl_warmup' |
| `--device` | The CUDA device to use. | 0 |
| `--seed` | The random seed for reproducibility. | 0 |
| `--L` | The number of samples for the MIS-ELBO. | 1 |
| `--L_final` | The number of samples for the final evaluation. | 5000 |
| `--lr` | The learning rate for the optimizer. | 0.0005 |
| `--dataset` | The dataset to use ('mnist' or 'fashion_mnist'). | 'mnist' |
| `--n_A` | Corresponds to 'S' in the paper (number of active components). | 1 |
| `--res_enc` | Whether to use a residual encoder. | 1 |
| `--estimator` | The estimator to use ('s2s' or 's2a'). | 's2s' |
| `--no_epochs` | The number of training epochs. | 2000 |
| `--num_workers`| The number of workers for the DataLoader. | 0 |
| `--compile_model`| Set to `1` to enable `torch.compile`. | 0 |
| `--prepare_data`| Set to `1` to run the data preparation script. | 0 |
