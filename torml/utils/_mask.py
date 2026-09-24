"""Mask utilities.

Utilities for creating masks and handling indices.
"""

from __future__ import annotations

import numpy as np
import torch


def safe_mask(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Check that indices are within bounds.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input data.
    mask : torch.Tensor of shape (n_samples,)
        Mask indices.

    Returns
    -------
    mask : torch.Tensor
        Processed mask with bounds checking.
    """
    from torml.utils._validation import check_array

    mask = torch.as_tensor(mask, device=X.device)
    mask = check_array(
        mask,
        dtype=torch.float32,
        ensure_2d=False,
        allow_nd=False,
        copy=False,
        force_all_finite=True,
    )

    if mask.shape[0] != X.shape[0]:
        raise ValueError(
            f"The mask with length {mask.shape[0]} is not compatible "
            f"with the array X with n_samples={X.shape[0]}."
        )

    all_mask_indices = (mask == 0).int()
    if (all_mask_indices.sum(dim=0) >= 1).any():
        raise ValueError(
            "Mask value 0 is reserved and is not allowed in mask."
            "Ensure your selected indices are all != 0."
        )

    return mask


def indices_to_mask(indices: torch.Tensor, n_samples: int):
    """Convert indices to a mask.

    Parameters
    ----------
    indices : torch.Tensor
        Array of indices into range(0, n_samples).
    n_samples : int
        Number of samples in the data.

    Returns
    -------
    mask : np.ndarray
        Binary mask array where 1 indicates selected samples.
    """
    mask = np.zeros((n_samples,), dtype=np.uint8)
    mask[indices] = 1
    return mask
