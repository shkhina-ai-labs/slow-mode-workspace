"""
SlowModeProjector — the shared low-rank Koopman-invariant projector S.

Three constraints, applied as soft regularizers (added to the loss) so the
projector is end-to-end trainable:

  (1) Stiefel orthogonality:  ||S^T S - I||_F^2  -> 0
  (2) Scale-equivariance under multiplicative dilation (anomalous dim Δ).
      S(λ * x) ≈ λ^Δ * S(x)  -- penalized at random λ ∈ [0.5, 2.0].
  (3) Contractive propagator: spectral_radius(K = S^T A S) < 1
      Implemented by penalizing operator norm exceeding 1.

These match the Round 2 narrowed claim (synthesis 04b) — the constraints that
distinguish SMW from a Gram-matrix-based latent-memory operator.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SlowModeProjector(nn.Module):
    """
    Single low-rank Koopman-invariant projector S : R^d -> R^{d_w}.

    Args:
        d_model: residual stream width (e.g. 4096)
        d_w:     workspace width, d_w << d_model (e.g. 64)
        anomalous_dim: target scaling exponent Δ in S(λx) ≈ λ^Δ S(x).
                       Default 1.0 (linear scale-equivariance).

    The same instance is meant to be SHARED across blocks (weight-tied).
    """

    def __init__(self, d_model: int, d_w: int, anomalous_dim: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.d_w = d_w
        self.anomalous_dim = anomalous_dim

        # Initialize with semi-orthogonal columns (Stiefel start).
        weight = torch.empty(d_model, d_w)
        nn.init.orthogonal_(weight)
        self.S = nn.Parameter(weight)

        # Learnable Koopman propagator K acting in workspace space.
        # Spectral radius is bounded < 1 via the regularizer.
        K = torch.empty(d_w, d_w)
        nn.init.orthogonal_(K)
        # Slight contraction at init.
        self.K = nn.Parameter(K * 0.9)

    # ---- forward operations --------------------------------------------------

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project residual stream into the workspace.  x: (B, T, d) -> (B, T, d_w)."""
        return x @ self.S

    def lift(self, w: torch.Tensor) -> torch.Tensor:
        """Lift workspace back to residual stream.  w: (B, T, d_w) -> (B, T, d)."""
        # Under Stiefel S^T S = I, the natural lift is S (since S^T is the projector).
        return w @ self.S.T

    def propagate(self, w: torch.Tensor) -> torch.Tensor:
        """Apply the Koopman propagator inside the workspace."""
        return w @ self.K.T

    # ---- regularizers --------------------------------------------------------

    def stiefel_loss(self) -> torch.Tensor:
        """Enforce S^T S ≈ I (commitment 1)."""
        StS = self.S.T @ self.S
        I = torch.eye(self.d_w, device=self.S.device, dtype=self.S.dtype)
        return ((StS - I) ** 2).sum()

    def scale_equivariance_loss(self, x: torch.Tensor, n_lambdas: int = 4) -> torch.Tensor:
        """
        Enforce S(λx) ≈ λ^Δ S(x) for random λ ∈ [0.5, 2.0]  (commitment 2).
        x: (B, T, d_model)
        """
        # Sample log-uniform λ in [0.5, 2.0].
        log_lo, log_hi = torch.log(torch.tensor(0.5)), torch.log(torch.tensor(2.0))
        u = torch.rand(n_lambdas, device=x.device)
        lambdas = torch.exp(log_lo + u * (log_hi - log_lo))  # (n_lambdas,)

        Sx = self.project(x)  # (B, T, d_w)
        loss = 0.0
        for lam in lambdas:
            S_lamx = self.project(lam * x)
            target = (lam ** self.anomalous_dim) * Sx
            loss = loss + ((S_lamx - target) ** 2).mean()
        return loss / n_lambdas

    def contractivity_loss(self) -> torch.Tensor:
        """
        Penalize spectral radius of K exceeding 1 (commitment 3).

        Uses operator 2-norm (≥ spectral radius) as a tractable upper bound:
        ||K||_2 = largest singular value.  Penalty:  max(0, ||K||_2 - 1)^2.
        """
        sigma_max = torch.linalg.matrix_norm(self.K, ord=2)
        return torch.relu(sigma_max - 1.0) ** 2

    def total_regularizer(
        self,
        x: torch.Tensor | None = None,
        w_stiefel: float = 1e-2,
        w_scale: float = 1e-3,
        w_contract: float = 1e-2,
    ) -> torch.Tensor:
        """
        Sum of the three constraint losses, weighted.

        Pass `x` (a residual-stream activation tensor) to include the
        scale-equivariance term.  Without `x`, only Stiefel + contractivity.
        """
        loss = w_stiefel * self.stiefel_loss() + w_contract * self.contractivity_loss()
        if x is not None:
            loss = loss + w_scale * self.scale_equivariance_loss(x)
        return loss

    # ---- diagnostic probes ---------------------------------------------------

    @torch.no_grad()
    def eigenvalue_spectrum(self) -> torch.Tensor:
        """Eigenvalues of K (complex), for the slow-band probe (P1)."""
        return torch.linalg.eigvals(self.K)

    @torch.no_grad()
    def slow_band_mass(self, threshold: float = 0.95) -> torch.Tensor:
        """
        Fraction of eigenvalues with |λ| ≥ threshold.

        For a healthy SMW, this should be > 0.3 after training (slow modes
        concentrate near the unit circle).  P1 diagnostic.
        """
        evals = self.eigenvalue_spectrum()
        return (evals.abs() >= threshold).float().mean()
