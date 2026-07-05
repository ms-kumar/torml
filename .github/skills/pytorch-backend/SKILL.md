---
name: pytorch-backend
description: 'Use when: implementing torml algorithms with PyTorch tensors, torch.linalg, vectorization, dtype/device policy, autograd decisions, avoiding NumPy/SciPy backends.'
argument-hint: '<algorithm or tensor operation>'
---

# PyTorch Backend

Use this skill when implementing numerical logic in torml.

## Backend Rules

1. Use `torch.Tensor` as the internal numerical representation.
2. Use `torch.linalg` for decompositions, solves, SVD, eigenvalues, and norms.
3. Prefer vectorized tensor operations over Python loops.
4. Avoid NumPy/SciPy in algorithm internals unless explicitly approved.
5. Keep CPU as the default device.

## Autograd Policy

- Prefer closed-form or explicit iterative algorithms where classical ML expects them.
- Use autograd only when it improves clarity or correctness.
- Keep autograd implementation details inside `fit`.
- Detach stored learned attributes when appropriate.

## Dtype and Device

- Default dtype: `torch.float32`.
- Preserve user-provided tensor device when possible.
- If an estimator accepts `device`, document it and test it.
