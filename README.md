# torml

`torml` is a scikit-learn-style machine learning library implemented from
scratch with [PyTorch](https://pytorch.org/) (`torch.Tensor`, `torch.linalg`)
as its numerical backend, instead of NumPy/SciPy.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import torch
from torml.linear_model import LinearRegression

X = torch.randn(50, 3)
y = X @ torch.tensor([1.0, 2.0, -1.0]) + 0.1

model = LinearRegression().fit(X, y)
model.predict(X[:5])
```

## Documentation

- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Coding guidelines: [`TORML_CODING_GUIDELINES.md`](TORML_CODING_GUIDELINES.md)
- Release notes: [`RELEASES.md`](RELEASES.md)

## License

MIT — see [`LICENSE`](LICENSE).
