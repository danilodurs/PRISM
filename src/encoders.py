"""Baseline (PCA) and prior-masked linear autoencoders, shared across every
prior tested by this harness.

Both are fit as autoencoders (encoder + linear decoder, reconstruction loss)
on training-fold profiles only, so the encoder fit itself respects the
donor-level split, not just the downstream probe. Kept deliberately linear
and shallow -- a single graph-constrained linear layer gives the prior's
structure the entire burden of the comparison, which is the point of the
experiment.

The masked encoder's sign handling is a HARD multiplicative constraint
(`weight = sign * softplus(magnitude) * mask`), not a soft initialization:
TRACE (the prior generalization of which this harness is built on) found
that a soft `sign * 0.1` initialization is fully overwritten by Adam within
the training budget, making a sign-scrambled control statistically
indistinguishable from the real graph -- the sign channel was never actually
enforced. Locking it as a hard constraint here means the same architecture
serves both signed and unsigned priors: unsigned priors simply pass
`sign = +1` everywhere a mask entry is nonzero, which still enforces
non-negative weights per edge but carries no directional information to
lock.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.decomposition import PCA


class PCAEncoder:
    """Baseline: linear, no graph structure. Output dim matched to the prior
    encoder's hidden-unit count.
    """

    def __init__(self, n_components: int, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        # PCA can't extract more components than min(n_samples, n_features), unlike
        # the prior encoder's fixed-width masked linear layer -- this only bites if
        # the training set shrinks below the hidden-unit-count-matched embedding dim.
        n_components = min(self.n_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.seed)
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "call fit() before transform()"
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.pca.transform(X)


class _MaskedLinearAutoencoder(torch.nn.Module):
    """weight = sign * softplus(raw_magnitude) * mask. softplus is always >= 0,
    so Adam can move the magnitude of an edge but can never flip a positive
    (activating) edge negative or vice versa -- the sign is a hard constraint
    on the parameterization, not an initial value the optimizer is free to
    overwrite.
    """

    def __init__(self, mask: np.ndarray, sign: np.ndarray):
        super().__init__()
        n_genes, n_units = mask.shape
        # softplus(init_raw) ~= 0.1, a small, stable initial edge magnitude
        init_raw = math.log(math.exp(0.1) - 1.0)
        self.raw_magnitude = torch.nn.Parameter(torch.full((n_genes, n_units), init_raw, dtype=torch.float32))
        self.register_buffer("sign", torch.tensor(sign, dtype=torch.float32))
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.encoder_bias = torch.nn.Parameter(torch.zeros(n_units))
        self.decoder = torch.nn.Linear(n_units, n_genes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sign * torch.nn.functional.softplus(self.raw_magnitude) * self.mask
        return torch.tanh(x @ w + self.encoder_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)


class PriorEncoder:
    """Prior-aware encoder: one output unit per hidden unit (TF, anchor gene,
    etc. depending on the active prior), reading only its graph-defined linked
    genes with a learnable, hard-sign-constrained weight. Used for both a
    prior's real graph and every structural control derived from it -- only
    the (mask, sign) pair changes; the architecture is held fixed.
    """

    def __init__(
        self,
        mask: np.ndarray,
        sign: np.ndarray,
        seed: int = 0,
        n_epochs: int = 80,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        patience: int = 15,
        min_delta: float = 1e-4,
    ):
        self.mask = mask
        self.sign = sign
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.model: _MaskedLinearAutoencoder | None = None
        self._mu = None
        self._sigma = None

    def fit(self, X: np.ndarray) -> "PriorEncoder":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._mu = X.mean(axis=0, keepdims=True)
        self._sigma = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self._mu) / self._sigma

        self.model = _MaskedLinearAutoencoder(self.mask, self.sign)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_t = torch.tensor(Xn, dtype=torch.float32)

        # early-stop on reconstruction-loss plateau: this is a small linear model,
        # so it converges in tens of epochs and running the full epoch budget
        # regardless just burns CPU time across hundreds of CV fits.
        self.model.train()
        best_loss = float("inf")
        epochs_since_improvement = 0
        for _ in range(self.n_epochs):
            opt.zero_grad()
            recon = self.model(x_t)
            loss = torch.mean((recon - x_t) ** 2)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            if loss_val < best_loss - self.min_delta:
                best_loss = loss_val
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    break
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit() before transform()"
        Xn = (X - self._mu) / self._sigma
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(torch.tensor(Xn, dtype=torch.float32))
        return z.numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
