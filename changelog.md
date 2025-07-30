# Changelog

## [Unreleased] - 2025-07-31

### Changed

- **Performance:** Refactored `models/misvae.py` to improve performance without affecting results.
    - Replaced expensive `Normal` distribution object creation in `compute_prior` with a more efficient, direct calculation of the standard normal log-probability density function.
    - Replaced all `np.log` calls with `torch.log` to ensure all calculations remain within the PyTorch ecosystem, avoiding unnecessary data transfers between CPU and the target device.

- **Refactor:** Improved code quality and readability in `models/encoders.py`.
    - In `GatedConv2dResidualEncoder`, eliminated duplicate code for `mu` and `std` calculation by introducing a `_residual_block` helper method. This improves maintainability and reduces code size.

- **Chore:** Commented out unused imports (`pdb` in `misvae.py` and `torch.nn.functional` in `encoders.py`) to clean up the code.
