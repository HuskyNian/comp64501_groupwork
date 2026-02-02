from dataclasses import dataclass
import numpy as np
import os
from datetime import datetime
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, RBF, RationalQuadratic, WhiteKernel
from ShallowPocketModel import QuantumChannelAnalyzer as q
import matplotlib.pyplot as plt
from bo_utils import suppress_warnings

def main():
    suppress_warnings()

    analyzer = q(g=0.8, r=0.3, t=5.0, T=5.0)
    choi_superchannel = analyzer.create_shallow_pocket_model()

    # --- GP (Bayesian Optimization) to maximize F_U over (theta, phi, psi) ---
    @dataclass(frozen=True)
    class Bounds:
        low: float
        high: float

    # Standard Z-Y-Z Euler angle bounds for SU(2): theta ∈ [0, 2π), phi ∈ [0, π], psi ∈ [0, 2π)
    bounds = (Bounds(0.0, 2 * np.pi), Bounds(0.0, np.pi), Bounds(0.0, 2 * np.pi))

    def _clip_to_bounds(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        for i, b in enumerate(bounds):
            x[i] = float(np.clip(x[i], b.low, b.high))
        return x

    eval_cache: dict[tuple[float, float, float], float] = {}

    def fidelity_from_angles(theta: float, phi: float, psi: float) -> float:
        # Cache by rounded params to avoid duplicate expensive evals
        key = (round(float(theta), 10), round(float(phi), 10), round(float(psi), 10))
        if key in eval_cache:
            return eval_cache[key]
        try:
            V = analyzer.parameterised_unitary(theta, phi, psi)
            choi_input = analyzer.choi_state_unitary(V)
            choi_output = analyzer.link_product(choi_superchannel, choi_input)
            _, F_U = analyzer.closest_unitary_channel(choi_output)
            val = float(np.real(F_U))
        except Exception:
            # If something goes wrong numerically, treat as bad candidate.
            val = 0.0
        # Fidelity should live in [0, 1], but keep it safe.
        val = float(np.clip(val, 0.0, 1.0))
        eval_cache[key] = val
        return val

    def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        # EI for maximization: E[max(0, f(x) - y_best - xi)]
        sigma = np.maximum(sigma, 1e-12)
        imp = mu - y_best - xi
        Z = imp / sigma
        return imp * norm.cdf(Z) + sigma * norm.pdf(Z)

    def gp_optimize(
        kernel,
        kernel_name: str,
        n_init: int = 10,
        n_iter: int = 35,
        n_candidates: int = 3000,
        seed: int = 0,
        verbose: bool = True,
        X_init: np.ndarray | None = None,
        Xcand_list: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        def sample_uniform(n: int) -> np.ndarray:
            X = np.zeros((n, 3), dtype=float)
            for i, b in enumerate(bounds):
                X[:, i] = rng.uniform(b.low, b.high, size=n)
            return X

        # Initial design: start from (0, 0, 0), then fill the rest with random samples
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        if X_init is None:
            X = np.zeros((n_init, 3), dtype=float)
            X[0, :] = np.array([0.0, 0.0, 0.0], dtype=float)
            if n_init > 1:
                X[1:, :] = sample_uniform(n_init - 1)
        else:
            X = np.asarray(X_init, dtype=float).copy()
            if X.shape != (n_init, 3):
                raise ValueError(f"X_init must have shape {(n_init, 3)}; got {X.shape}")
        y_list: list[float] = []
        if verbose:
            print(f"\n=== GP Optimization (maximize F_U) | kernel={kernel_name} ===")
            print(f"init points: {n_init}, BO iters: {n_iter}, candidates/iter: {n_candidates}, seed: {seed}")
            print("\n--- Initial random evaluations ---")
        for i in range(n_init):
            theta_i, phi_i, psi_i = (float(X[i, 0]), float(X[i, 1]), float(X[i, 2]))
            yi = fidelity_from_angles(theta_i, phi_i, psi_i)
            y_list.append(yi)
            if verbose:
                print(f"init {i+1:02d}/{n_init}: theta={theta_i:.6f}, phi={phi_i:.6f}, psi={psi_i:.6f} -> F_U={yi:.6f}")
        y = np.array(y_list, dtype=float)
        best_so_far = np.maximum.accumulate(y)

        # GP model
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=3)

        if verbose:
            print("\n--- Bayesian optimization iterations ---")
        for it in range(n_iter):
            gp.fit(X, y)
            y_best = float(np.max(y))

            # Optimize acquisition on a random candidate set (works well in 3D)
            if Xcand_list is None:
                Xcand = sample_uniform(n_candidates)
            else:
                Xcand = Xcand_list[it]
            mu, std = gp.predict(Xcand, return_std=True)
            ei = expected_improvement(mu, std, y_best=y_best, xi=0.01)
            x_next = _clip_to_bounds(Xcand[int(np.argmax(ei))])

            y_next = fidelity_from_angles(*x_next)
            X = np.vstack([X, x_next[None, :]])
            y = np.append(y, y_next)
            best_so_far = np.append(best_so_far, float(np.max(y)))
            if verbose:
                theta_n, phi_n, psi_n = (float(x_next[0]), float(x_next[1]), float(x_next[2]))
                best_now = float(np.max(y))
                print(
                    f"iter {it+1:02d}/{n_iter}: theta={theta_n:.6f}, phi={phi_n:.6f}, psi={psi_n:.6f} "
                    f"-> F_U={float(y_next):.6f} | best={best_now:.6f}"
                )

        best_idx = int(np.argmax(y))
        return X[best_idx], float(y[best_idx]), X, y, best_so_far

    def _sanitize(name: str) -> str:
        return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)

    def save_run_artifacts(
        out_dir: str,
        tag: str,
        kernel_name: str,
        X_hist: np.ndarray,
        y_hist: np.ndarray,
        best_hist: np.ndarray,
        n_init: int,
    ) -> tuple[str, str]:
        kernel_slug = _sanitize(kernel_name)
        png_path = os.path.join(out_dir, f"gp_fidelity_{tag}_{kernel_slug}.png")
        csv_path = os.path.join(out_dir, f"gp_fidelity_{tag}_{kernel_slug}.csv")

        eval_idx = np.arange(1, y_hist.shape[0] + 1, dtype=int)
        csv_mat = np.column_stack([eval_idx, y_hist, best_hist, X_hist])
        np.savetxt(
            csv_path,
            csv_mat,
            delimiter=",",
            header="eval_idx,fidelity,best_so_far,theta,phi,psi",
            comments="",
        )

        plt.figure(figsize=(8, 4.5))
        plt.plot(eval_idx, y_hist, label="observed F_U", alpha=0.35, linewidth=1.5)
        plt.plot(eval_idx, best_hist, label="best so far", linewidth=2.2)
        plt.axvline(x=n_init, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="end of init")
        plt.title(f"GP BO Fidelity Trace | {kernel_name}")
        plt.xlabel("Evaluation")
        plt.ylabel("Fidelity F_U")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        return png_path, csv_path

    # --- Run multiple kernels, save each plot, and save a combined comparison plot ---
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_init = 10
    n_iter = 100
    n_candidates = 2500
    seed = 1

    # For a fair comparison, reuse the same initial points and the same candidate sets each iteration.
    rng_shared = np.random.default_rng(seed)
    X_init = np.zeros((n_init, 3), dtype=float)
    X_init[0, :] = np.array([0.0, 0.0, 0.0], dtype=float)
    if n_init > 1:
        for dim, b in enumerate(bounds):
            X_init[1:, dim] = rng_shared.uniform(b.low, b.high, size=(n_init - 1))

    Xcand_list: list[np.ndarray] = []
    for _ in range(n_iter):
        Xcand = np.zeros((n_candidates, 3), dtype=float)
        for dim, b in enumerate(bounds):
            Xcand[:, dim] = rng_shared.uniform(b.low, b.high, size=n_candidates)
        Xcand_list.append(Xcand)

    # Kernel candidates (all include a small WhiteKernel for numerical stability/noise)
    kernels: list[tuple[str, object]] = [
        (
            "RBF(ARD)",
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "Matern(nu=1.5, ARD)",
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "Matern(nu=2.5, ARD)",
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "RationalQuadratic",
            ConstantKernel(1.0, (1e-3, 1e3))
            * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "ExpSineSquared(periodic)",
            ConstantKernel(1.0, (1e-3, 1e3))
            * ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
    ]

    results: dict[str, dict[str, object]] = {}
    best_overall = (-1.0, None, None)  # (best_f, kernel_name, best_x)

    for kname, kern in kernels:
        best_x, best_f, X_hist, y_hist, best_hist = gp_optimize(
            kernel=kern,
            kernel_name=kname,
            n_init=n_init,
            n_iter=n_iter,
            n_candidates=n_candidates,
            seed=seed,
            verbose=True,
            X_init=X_init,
            Xcand_list=Xcand_list,
        )

        png_path, csv_path = save_run_artifacts(out_dir, tag, kname, X_hist, y_hist, best_hist, n_init=n_init)
        print(f"\nSaved plot for kernel '{kname}' to: {png_path}")
        print(f"Saved CSV  for kernel '{kname}' to: {csv_path}")

        results[kname] = {
            "best_x": best_x,
            "best_f": best_f,
            "X_hist": X_hist,
            "y_hist": y_hist,
            "best_hist": best_hist,
            "png_path": png_path,
            "csv_path": csv_path,
        }
        if float(best_f) > best_overall[0]:
            best_overall = (float(best_f), kname, best_x)

    # Combined comparison plot (best-so-far traces)
    compare_png = os.path.join(out_dir, f"gp_fidelity_{tag}_comparison.png")
    plt.figure(figsize=(9.5, 5.0))
    for kname, payload in results.items():
        best_hist = np.asarray(payload["best_hist"], dtype=float)
        eval_idx = np.arange(1, best_hist.shape[0] + 1, dtype=int)
        plt.plot(eval_idx, best_hist, linewidth=2.0, label=kname)
    plt.axvline(x=n_init, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("GP BO Fidelity Comparison (best-so-far)")
    plt.xlabel("Evaluation")
    plt.ylabel("Best-so-far Fidelity F_U")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(compare_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison plot to: {compare_png}")

    # Pick best overall kernel run for the final channel/unitary demo
    best_f, best_kernel, best_x = best_overall
    assert best_kernel is not None and best_x is not None
    theta_opt, phi_opt, psi_opt = (float(best_x[0]), float(best_x[1]), float(best_x[2]))

    print("\n=== Best Result Across Kernels ===")
    print(f"best kernel = {best_kernel}")
    print(f"best theta  = {theta_opt:.6f}, phi = {phi_opt:.6f}, psi = {psi_opt:.6f}")
    print(f"best F_U    = {best_f:.6f}")

    # --- Show the channels/unitaries at the optimized angles (same as original demo) ---
    V = analyzer.parameterised_unitary(theta_opt, phi_opt, psi_opt)
    print("\nParameterized unitary (optimized, rounded):")
    print(analyzer.qobj_round(V, 2))

    choi_input = analyzer.choi_state_unitary(V)
    print("\nInput channel (optimized, rounded):")
    print(analyzer.qobj_round(choi_input, 2))

    choi_output = analyzer.link_product(choi_superchannel, choi_input)
    print("\nOutput channel (optimized, rounded):")
    print(analyzer.qobj_round(choi_output, 2))

    U, F_U = analyzer.closest_unitary_channel(choi_output)
    print("\nClosest unitary U (optimized, rounded):")
    print(analyzer.qobj_round(U, 2))
    print(f"Fidelity (Closest Unitary vs Output): {float(F_U):.6f}")

if __name__ == "__main__":
    main()