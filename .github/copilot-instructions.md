# Copilot Instructions for torml

torml follows scikit-learn engineering conventions while using PyTorch as the backend.

## Always follow

- Use `TORML_CODING_GUIDELINES.md` and `AGENTS.md` as primary project instructions.
- Implement algorithms from scratch using `torch`; do not call scikit-learn internals.
- Follow scikit-learn-style estimator APIs: `fit`, `predict`, `transform`, `score`, `get_params`, `set_params`.
- Store fitted attributes with trailing underscores.
- Add tests and documentation updates for public behavior changes.

## Avoid

- NumPy/SciPy algorithm implementations unless explicitly requested.
- Import-time side effects.
- Deep-learning-style APIs for classical estimators unless they are internal implementation details.
