from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

import numpy as np


def suppress_warnings() -> None:
    """
    Silence common noisy warnings seen in this project (sklearn GP convergence + qutip user warnings).
    Call this early in scripts/notebooks.
    """
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        # sklearn might not be installed in some environments
        pass

    # qutip sometimes warns about missing matplotlib, etc.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_outputs_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sanitize_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """EI acquisition for maximization."""
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-12)
    imp = mu - y_best - xi
    Z = imp / sigma
    return imp * norm.cdf(Z) + sigma * norm.pdf(Z)


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


def sample_uniform(rng: np.random.Generator, bounds: Iterable[Bounds], n: int) -> np.ndarray:
    b = list(bounds)
    X = np.zeros((n, len(b)), dtype=float)
    for i, bi in enumerate(b):
        X[:, i] = rng.uniform(bi.low, bi.high, size=n)
    return X


def clip_to_bounds(x: np.ndarray, bounds: Iterable[Bounds]) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for i, b in enumerate(bounds):
        x[i] = float(np.clip(x[i], b.low, b.high))
    return x


def save_trace_csv(
    csv_path: str,
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    best_hist: np.ndarray,
    header_cols: list[str],
) -> None:
    eval_idx = np.arange(1, y_hist.shape[0] + 1, dtype=int)
    mat = np.column_stack([eval_idx, y_hist, best_hist, X_hist])
    header = ",".join(["eval_idx", "fidelity", "best_so_far"] + header_cols)
    np.savetxt(csv_path, mat, delimiter=",", header=header, comments="")


def save_trace_plot(
    png_path: str,
    y_hist: np.ndarray,
    best_hist: np.ndarray,
    n_init: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    idx = np.arange(1, y_hist.shape[0] + 1, dtype=int)
    plt.figure(figsize=(8, 4.5))
    plt.plot(idx, y_hist, label="observed F_U", alpha=0.35, linewidth=1.5)
    plt.plot(idx, best_hist, label="best so far", linewidth=2.2)
    plt.axvline(x=n_init, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="end of init")
    plt.title(title)
    plt.xlabel("Evaluation")
    plt.ylabel("Fidelity F_U")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


