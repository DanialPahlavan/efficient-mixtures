# Changelog

## [Unreleased] - 2025-07-31

### Changed

- **Performance:** Drastically improved the performance of `EnsembleGatedConv2dEncoders` for large numbers of components (`S`).
    - The `forward` method in `models/encoders.py` was completely refactored to use a highly efficient tensor manipulation strategy.
    - Instead of creating large, memory-intensive intermediate tensors of shape `(batch_size, S, S)`, the new implementation uses `repeat_interleave` and `repeat` to directly construct the final flattened tensor required by the linear layers.
    - This change provides a significant speed-up and reduces memory consumption, especially when `S` is large, without altering the final output.

- **Performance:** Refactored `models/misvae.py` to improve performance by pre-calculating and buffering constants.
    - In the `__init__` method, `log(2*pi)`, `log(S)`, and `log(n_A)` are now computed once and registered as buffers.
    - This avoids repeated and unnecessary calculations of these values in every forward pass during training, leading to a significant speed-up.
    - The `compute_prior` and `get_log_w` methods were updated to use these pre-computed buffers.

- **Performance:** Refactored `models/misvae.py` to improve performance without affecting results.
    - Replaced expensive `Normal` distribution object creation in `compute_prior` with a more efficient, direct calculation of the standard normal log-probability density function.
    - Replaced all `np.log` calls with `torch.log` to ensure all calculations remain within the PyTorch ecosystem, avoiding unnecessary data transfers between CPU and the target device.

- **Refactor:** Improved code quality and readability in `models/encoders.py`.
    - In `GatedConv2dResidualEncoder`, eliminated duplicate code for `mu` and `std` calculation by introducing a `_residual_block` helper method. This improves maintainability and reduces code size.

- **Performance:** Optimized the training loop in `mnist_train.py`.
    - Pre-calculated the `multinomial_probs` tensor outside the main loop in the `trainer` function to prevent its recreation in every batch, reducing overhead.

- **Fix:** Corrected a minor issue in `mnist_train.py` by wrapping a scalar value with `np.array()` before passing it to `np.save` to ensure type consistency.

- **Chore:** Commented out unused imports (`pdb` in `misvae.py` and `torch.nn.functional` in `encoders.py`) to clean up the code.
