from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF, RationalQuadratic, WhiteKernel

from bo_utils import Bounds, clip_to_bounds, ensure_outputs_dir, expected_improvement, sanitize_filename, save_trace_csv, save_trace_plot, suppress_warnings, timestamp_tag
from DoubleSlot import QuantumChannelAnalyzer as DoubleSlotAnalyzer
from SingleSlot import QuantumChannelAnalyzer as SingleSlotAnalyzer


def _prepare_run_dirs(out_dir: str | None, mode_prefix: str, tag: str | None = None) -> tuple[str, str]:
    """
    Create per-run output directories:
      outputs/<timestamp>/<mode_prefix>/
    Returns (mode_dir, run_root_dir).
    """
    base_dir = os.path.dirname(__file__)
    outputs_dir = ensure_outputs_dir(out_dir or base_dir)
    tag = tag or timestamp_tag()
    run_root = os.path.join(outputs_dir, tag)
    mode_dir = os.path.join(run_root, mode_prefix)
    os.makedirs(mode_dir, exist_ok=True)
    return mode_dir, run_root


def _write_params_txt(
    mode_dir: str,
    *,
    mode: str,
    g,
    r: float,
    t: float,
    n_init: int,
    n_iter: int,
    n_candidates: int,
    seeds: list[int],
    phi_in_pi: bool,
    xi: float,
    note: str | None = None,
) -> str:
    fp = os.path.join(mode_dir, "params.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"mode: {mode}\n")
        f.write(f"g: {g}\n")
        f.write(f"r: {r}\n")
        f.write(f"t: {t}\n")
        f.write(f"n_init: {n_init}\n")
        f.write(f"n_iter: {n_iter}\n")
        f.write(f"n_candidates: {n_candidates}\n")
        f.write(f"seeds: {seeds}\n")
        f.write(f"phi_in_pi: {phi_in_pi}\n")
        f.write(f"xi: {xi}\n")
        if note:
            f.write(f"note: {note}\n")
    return fp


def _append_best_solutions(
    mode_dir: str,
    *,
    kernel_name: str,
    seeds: list[int],
    best_fs: list[float],
    best_xs: list[np.ndarray],
) -> None:
    fp = os.path.join(mode_dir, "params.txt")
    with open(fp, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"[best_solutions] kernel: {kernel_name}\n")
        for sd, bf, bx in zip(seeds, best_fs, best_xs):
            bx_list = [float(v) for v in np.asarray(bx).reshape(-1)]
            f.write(f"  seed={sd}: best_f={bf:.12f}, best_x={bx_list}\n")


def _double_slot_bounds(phi_in_pi: bool) -> tuple[Bounds, ...]:
    if phi_in_pi:
        # (theta, phi, psi) for each slot: theta,psi in [0,2pi], phi in [0,pi]
        return (
            Bounds(0.0, 2 * np.pi),
            Bounds(0.0, np.pi),
            Bounds(0.0, 2 * np.pi),
            Bounds(0.0, 2 * np.pi),
            Bounds(0.0, np.pi),
            Bounds(0.0, 2 * np.pi),
        )
    return tuple(Bounds(0.0, 2 * np.pi) for _ in range(6))


def _double_slot_bounds_reduced3(phi_in_pi: bool) -> tuple[Bounds, ...]:
    """
    Reduced 3-variable bounds when we force theta1=0, theta2=0, and psi2=0.
    Variable order: (phi1, psi1, phi2).
    """
    if phi_in_pi:
        return (
            Bounds(0.0, np.pi),      # phi1
            Bounds(0.0, 2 * np.pi),  # psi1
            Bounds(0.0, np.pi),      # phi2
        )
    return tuple(Bounds(0.0, 2 * np.pi) for _ in range(3))


def _expand_double_slot_3_to_6(x3: np.ndarray) -> np.ndarray:
    """
    Map reduced variables (phi1, psi1, phi2) -> full 6 variables
    (theta1=0, phi1, psi1, theta2=0, phi2, psi2=0).
    """
    x3 = np.asarray(x3, dtype=float).reshape(-1)
    if x3.shape[0] != 3:
        raise ValueError(f"Expected x3 with 3 elements, got {x3.shape}")
    phi1, psi1, phi2 = (float(v) for v in x3)
    return np.array([0.0, phi1, psi1, 0.0, phi2, 0.0], dtype=float)


def _kernels_for_dim(dim: int) -> list[tuple[str, object]]:
    """
    Shared kernel set for BO.
    Note: We use ConstantKernel * base + WhiteKernel across all kernels for consistent scaling and stability.
    """
    return [
        (
            "RBF_ARD",
            ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * dim, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "Matern15_ARD",
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * dim, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "Matern25_ARD",
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * dim, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "RationalQuadratic",
            ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
        (
            "Linear_DotProduct",
            ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2)),
        ),
    ]


def _run_bo_single_seed(
    *,
    objective,
    bounds: tuple[Bounds, ...],
    kernel_name: str,
    kernel_obj,
    n_init: int,
    n_iter: int,
    n_candidates: int,
    seed: int,
    xi: float,
    verbose: bool,
    do_local_refine: bool = False,
    local_refine_steps: int = 0,
    local_refine_sigma: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run BO for one kernel and one seed.
    Returns (X_hist, y_hist, best_so_far_hist) where len == n_init + n_iter.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    def sample_uniform(n: int) -> np.ndarray:
        X = np.zeros((n, dim), dtype=float)
        for i, b in enumerate(bounds):
            X[:, i] = rng.uniform(b.low, b.high, size=n)
        return X

    if n_init < 1:
        raise ValueError("n_init must be >= 1")

    # Initial design: x0=0 then random
    X = np.zeros((n_init, dim), dtype=float)
    X[0, :] = 0.0
    if n_init > 1:
        X[1:, :] = sample_uniform(n_init - 1)
    y = np.array([objective(row) for row in X], dtype=float)
    best_hist = np.maximum.accumulate(y)

    if verbose:
        print(f"\n--- Kernel={kernel_name} | seed={seed} ---")
        print(f"init: n_init={n_init}, n_iter={n_iter}, n_candidates={n_candidates}, xi={xi}")
        print(f"init best={float(best_hist[-1]):.6f}")

    gp = GaussianProcessRegressor(kernel=kernel_obj, normalize_y=True, random_state=seed, n_restarts_optimizer=5)

    for it in range(n_iter):
        gp.fit(X, y)
        y_best = float(np.max(y))
        Xcand = sample_uniform(n_candidates)
        mu, std = gp.predict(Xcand, return_std=True)
        ei = expected_improvement(mu, std, y_best=y_best, xi=xi)
        x_next = clip_to_bounds(Xcand[int(np.argmax(ei))], bounds)
        y_next = objective(x_next)
        X = np.vstack([X, x_next[None, :]])
        y = np.append(y, y_next)
        best_hist = np.append(best_hist, float(np.max(y)))
        if verbose and (it < 3 or (it + 1) % 20 == 0 or it == n_iter - 1):
            print(f"iter {it+1:03d}/{n_iter}: y_next={float(y_next):.6f} | best={float(best_hist[-1]):.6f}")

    # Optional local refinement: treat as additional evaluations appended to the trace
    if do_local_refine and local_refine_steps > 0:
        best_idx = int(np.argmax(y))
        x0 = X[best_idx].copy()
        f0 = float(y[best_idx])
        sigma = float(local_refine_sigma)
        if verbose:
            print(f"--- Local refine: steps={local_refine_steps}, sigma={sigma} ---")
        for j in range(local_refine_steps):
            x_try = x0 + rng.normal(0.0, sigma, size=dim)
            x_try = clip_to_bounds(x_try, bounds)
            y_try = float(objective(x_try))
            X = np.vstack([X, x_try[None, :]])
            y = np.append(y, y_try)
            if y_try > f0:
                x0, f0 = x_try, y_try
                if verbose and ((j + 1) % 200 == 0 or j < 5):
                    print(f"refine {j+1:04d}/{local_refine_steps}: improved -> best={f0:.6f}")
            best_hist = np.append(best_hist, float(np.max(y)))

    best_idx = int(np.argmax(y))
    best_x = X[best_idx]
    best_f = float(y[best_idx])
    return X, y, best_hist, best_x, best_f


def run_double_slot_bo(
    g: tuple[float, float, float] = (0.8, 1.0, 1.1), # all 0.8 
    r: float = 0.3, # 
    t: float = 5.0, # 10/3
    n_init: int = 15,
    n_iter: int = 80,
    n_candidates: int = 8000,
    seed: int = 1,
    verbose: bool = True,
    out_dir: str | None = None,
    # Search-space tweak: for SU(2) Euler Z-Y-Z, the middle angle phi is typically in [0, pi].
    phi_in_pi: bool = True,
    # EI exploration strength: larger xi explores more; smaller xi exploits more.
    xi: float = 0.01,
    # Local refinement: try to improve best_x after BO with random local search.
    do_local_refine: bool = False,
    local_refine_steps: int = 0,
    local_refine_sigma: float = 0.15,
    # Reduced 3-variable mode: force theta1=0, theta2=0, and psi2=0.
    reduced3_fix_th1_th2_ps2_zero: bool = False,
) -> Tuple[np.ndarray, float, str, str]:
    """
    DoubleSlot BO over 6 angles: (theta1,phi1,psi1,theta2,phi2,psi2) in [0, 2pi].
    If reduced3_fix_th1_th2_ps2_zero=True, we instead optimize 3 variables (phi1,psi1,phi2)
    with theta1=0, theta2=0, psi2=0 forced, but we still report/return the expanded 6-variable solution.
    Saves PNG+CSV and returns (best_x, best_f, png_path, csv_path).
    """
    suppress_warnings()

    analyzer = DoubleSlotAnalyzer(g=g, r=r, t=t)
    choi_superchannel = analyzer.create_shallow_pocket_model()

    bounds = _double_slot_bounds_reduced3(phi_in_pi) if reduced3_fix_th1_th2_ps2_zero else _double_slot_bounds(phi_in_pi)
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    cache: dict[tuple[float, ...], float] = {}

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        if reduced3_fix_th1_th2_ps2_zero:
            x = _expand_double_slot_3_to_6(x)
        key = tuple(round(float(v), 10) for v in x)
        if key in cache:
            return cache[key]
        try:
            th1, ph1, ps1, th2, ph2, ps2 = (float(v) for v in x)
            V1 = analyzer.parameterised_unitary(th1, ph1, ps1)
            V2 = analyzer.parameterised_unitary(th2, ph2, ps2)
            choi_input1 = analyzer.choi_state_unitary(V1)
            choi_input2 = analyzer.choi_state_unitary(V2)
            choi_output = analyzer.link_product(choi_superchannel, choi_input1, choi_input2)
            _, F_U = analyzer.closest_unitary_channel(choi_output)
            val = float(np.real(F_U))
        except Exception:
            val = 0.0
        val = float(np.clip(val, 0.0, 1.0))
        cache[key] = val
        return val

    if verbose and reduced3_fix_th1_th2_ps2_zero:
        print("\n[double_slot reduced3] Forcing theta1=0, theta2=0, psi2=0; optimizing (phi1,psi1,phi2).")

    kernels = _kernels_for_dim(3 if reduced3_fix_th1_th2_ps2_zero else 6)

    best_run_name = ""
    best_X = None
    best_y = None
    best_besthist = None
    best_f = -1.0
    for kname, kobj in kernels:
        Xk, yk, bestk, _, _ = _run_bo_single_seed(
            objective=objective,
            bounds=bounds,
            kernel_name=kname,
            kernel_obj=kobj,
            n_init=n_init,
            n_iter=n_iter,
            n_candidates=n_candidates,
            seed=seed,
            xi=xi,
            verbose=verbose,
        )
        f = float(np.max(yk))
        if f > best_f:
            best_f = f
            best_run_name = kname
            best_X, best_y, best_besthist = Xk, yk, bestk

    assert best_X is not None and best_y is not None and best_besthist is not None
    X, y, best_hist = best_X, best_y, best_besthist

    best_idx = int(np.argmax(y))
    best_x = X[best_idx]
    best_f = float(y[best_idx])

    # Optional local refinement: random Gaussian perturbations around the best point found.
    if do_local_refine and local_refine_steps > 0:
        if verbose:
            print("\n--- Local refine (random perturbations) ---")
        x0 = best_x.copy()
        f0 = best_f
        sigma = float(local_refine_sigma)
        for j in range(local_refine_steps):
            x_try = x0 + rng.normal(0.0, sigma, size=dim)
            x_try = clip_to_bounds(x_try, bounds)
            f_try = objective(x_try)
            if f_try > f0:
                x0, f0 = x_try, f_try
                if verbose:
                    print(f"refine {j+1:04d}/{local_refine_steps}: improved -> F_U={f0:.6f}, x={np.round(x0, 6)}")
        # update best if improved
        if f0 > best_f:
            best_x, best_f = x0, f0

    base_dir = os.path.dirname(__file__)
    out_dir = ensure_outputs_dir(out_dir or base_dir)
    tag = timestamp_tag()
    slug = sanitize_filename(f"double_slot_{'reduced3_' if reduced3_fix_th1_th2_ps2_zero else ''}{best_run_name}_phiPi{phi_in_pi}_xi{xi}")
    png_path = os.path.join(out_dir, f"{slug}_{tag}.png")
    csv_path = os.path.join(out_dir, f"{slug}_{tag}.csv")

    if reduced3_fix_th1_th2_ps2_zero:
        X_to_save = np.vstack([_expand_double_slot_3_to_6(row) for row in X])
        best_x_to_report = _expand_double_slot_3_to_6(best_x)
    else:
        X_to_save = X
        best_x_to_report = best_x

    save_trace_csv(
        csv_path=csv_path,
        X_hist=X_to_save,
        y_hist=y,
        best_hist=best_hist,
        header_cols=["theta1", "phi1", "psi1", "theta2", "phi2", "psi2"],
    )
    save_trace_plot(
        png_path=png_path,
        y_hist=y,
        best_hist=best_hist,
        n_init=n_init,
        title=f"DoubleSlot GP BO Fidelity Trace | best kernel: {best_run_name}",
    )

    if verbose:
        print("\n=== DoubleSlot Best Result ===")
        print("best kernel =", best_run_name)
        if reduced3_fix_th1_th2_ps2_zero:
            print("reduced3: theta1=0, theta2=0, psi2=0 forced; optimized (phi1,psi1,phi2).")
        print("best x (theta1,phi1,psi1,theta2,phi2,psi2) =", np.round(best_x_to_report, 6))
        print("best F_U =", best_f)
        print("Saved PNG:", png_path)
        print("Saved CSV:", csv_path)

    return best_x_to_report, best_f, png_path, csv_path


def run_double_slot_kernel_seed_sweep(
    *,
    g: tuple[float, float, float] = (0.8, 1.0, 1.1),
    r: float = 0.3,
    t: float = 5.0,
    n_init: int = 15,
    n_iter: int = 80,
    n_candidates: int = 8000,
    seeds: list[int] | None = None,
    phi_in_pi: bool = True,
    xi: float = 0.01,
    out_dir: str | None = None,
    do_local_refine: bool = False,
    local_refine_steps: int = 0,
    local_refine_sigma: float = 0.15,
    # Reduced 3-variable mode: force theta1=0, theta2=0, and psi2=0.
    reduced3_fix_th1_th2_ps2_zero: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    For each kernel, run BO across multiple seeds and plot a mean convergence curve with error bars.
    Saves one plot per kernel: mean(best-so-far) with +/- std shading.
    Returns a dict with raw traces per kernel.
    """
    suppress_warnings()
    seeds = seeds or [1, 2, 3, 4, 5]

    analyzer = DoubleSlotAnalyzer(g=g, r=r, t=t)
    choi_superchannel = analyzer.create_shallow_pocket_model()
    bounds = _double_slot_bounds_reduced3(phi_in_pi) if reduced3_fix_th1_th2_ps2_zero else _double_slot_bounds(phi_in_pi)
    kernels = _kernels_for_dim(3 if reduced3_fix_th1_th2_ps2_zero else 6)

    mode_dir, run_root = _prepare_run_dirs(out_dir, "double_slot")
    _write_params_txt(
        mode_dir,
        mode="double_slot",
        g=g,
        r=r,
        t=t,
        n_init=n_init,
        n_iter=n_iter,
        n_candidates=n_candidates,
        seeds=seeds,
        phi_in_pi=phi_in_pi,
        xi=xi,
        note=(
            "Kernel sweep with 5 seeds; convergence plots are mean ± std. "
            + ("[reduced3: theta1=0, theta2=0, psi2=0]" if reduced3_fix_th1_th2_ps2_zero else "[full6]")
        ),
    )

    results: Dict[str, Dict[str, object]] = {}

    for kname, kobj in kernels:
        out_kname = ("reduced3_" + kname) if reduced3_fix_th1_th2_ps2_zero else kname
        traces: List[np.ndarray] = []
        best_xs: List[np.ndarray] = []
        best_fs: List[float] = []
        for sd in seeds:
            cache: dict[tuple[float, ...], float] = {}

            def objective(x: np.ndarray) -> float:
                x = np.asarray(x, dtype=float).reshape(-1)
                if reduced3_fix_th1_th2_ps2_zero:
                    x = _expand_double_slot_3_to_6(x)
                key = tuple(round(float(v), 10) for v in x)
                if key in cache:
                    return cache[key]
                try:
                    th1, ph1, ps1, th2, ph2, ps2 = (float(v) for v in x)
                    V1 = analyzer.parameterised_unitary(th1, ph1, ps1)
                    V2 = analyzer.parameterised_unitary(th2, ph2, ps2)
                    choi_input1 = analyzer.choi_state_unitary(V1)
                    choi_input2 = analyzer.choi_state_unitary(V2)
                    choi_output = analyzer.link_product(choi_superchannel, choi_input1, choi_input2)
                    _, F_U = analyzer.closest_unitary_channel(choi_output)
                    val = float(np.real(F_U))
                except Exception:
                    val = 0.0
                val = float(np.clip(val, 0.0, 1.0))
                cache[key] = val
                return val

            _, _, best_hist, best_x, best_f = _run_bo_single_seed(
                objective=objective,
                bounds=bounds,
                kernel_name=kname,
                kernel_obj=kobj,
                n_init=n_init,
                n_iter=n_iter,
                n_candidates=n_candidates,
                seed=sd,
                xi=xi,
                verbose=True,
                do_local_refine=do_local_refine,
                local_refine_steps=local_refine_steps,
                local_refine_sigma=local_refine_sigma,
            )
            traces.append(best_hist)
            best_xs.append(best_x)
            best_fs.append(float(best_f))

        # Stack: shape (n_seeds, T)
        T = n_init + n_iter + (local_refine_steps if do_local_refine else 0)
        mat = np.vstack([tr[:T] for tr in traces])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        # Save raw traces as CSV: columns are seeds
        csv_path = os.path.join(mode_dir, f"double_slot_{sanitize_filename(out_kname)}_seeds.csv")
        header = "eval_idx," + ",".join([f"seed_{sd}" for sd in seeds]) + ",mean,std"
        eval_idx = np.arange(1, T + 1, dtype=int)
        out_mat = np.column_stack([eval_idx, mat.T, mean, std])
        np.savetxt(csv_path, out_mat, delimiter=",", header=header, comments="")

        # Plot mean with error band
        import matplotlib.pyplot as plt

        png_path = os.path.join(mode_dir, f"double_slot_{sanitize_filename(out_kname)}_convergence.png")
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(eval_idx, mean, linewidth=2.4, label=f"{out_kname} mean")
        plt.fill_between(eval_idx, mean - std, mean + std, alpha=0.20, label="±1 std")
        plt.axvline(x=n_init, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="end of init")
        plt.axvline(x=n_init + n_iter, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="end of BO")
        plt.title(f"DoubleSlot BO convergence | {out_kname} | {len(seeds)} seeds")
        plt.xlabel("Evaluation")
        plt.ylabel("Best-so-far Fidelity $F_U$")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close()

        # Append best solutions (per seed) to params.txt in this mode folder
        _append_best_solutions(
            mode_dir,
            kernel_name=("reduced3_" + kname) if reduced3_fix_th1_th2_ps2_zero else kname,
            seeds=seeds,
            best_fs=best_fs,
            best_xs=[_expand_double_slot_3_to_6(x) if reduced3_fix_th1_th2_ps2_zero else x for x in best_xs],
        )

        results[kname] = {
            "seeds": seeds,
            "best_histories": traces,
            "best_fs": best_fs,
            "best_xs": best_xs,
            "mean": mean,
            "std": std,
            "csv_path": csv_path,
            "png_path": png_path,
            "mode_dir": mode_dir,
            "run_root": run_root,
        }

    return results


def run_single_slot_kernel_seed_sweep(
    *,
    g: tuple[float, float] = (0.8, 1.6),
    r: float = 0.3, # r 
    t: float = 10/3, # t 10/3 
    n_init: int = 15,
    n_iter: int = 80,
    n_candidates: int = 8000,
    seeds: list[int] | None = None,
    phi_in_pi: bool = True,
    xi: float = 0.01,
    out_dir: str | None = None,
    do_local_refine: bool = False,
    local_refine_steps: int = 0,
    local_refine_sigma: float = 0.15,
) -> Dict[str, Dict[str, object]]:
    """
    Single-slot (3 variables) version: run BO across multiple seeds and plot mean ± std convergence per kernel.
    """
    suppress_warnings()
    seeds = seeds or [1, 2, 3, 4, 5]

    analyzer = SingleSlotAnalyzer(g=g, r=r, t=t)
    choi_superchannel = analyzer.create_shallow_pocket_model()

    if phi_in_pi:
        bounds = (Bounds(0.0, 2 * np.pi), Bounds(0.0, np.pi), Bounds(0.0, 2 * np.pi))
    else:
        bounds = (Bounds(0.0, 2 * np.pi), Bounds(0.0, 2 * np.pi), Bounds(0.0, 2 * np.pi))

    kernels = _kernels_for_dim(3)

    mode_dir, run_root = _prepare_run_dirs(out_dir, "single_slot")
    _write_params_txt(
        mode_dir,
        mode="single_slot",
        g=g,
        r=r,
        t=t,
        n_init=n_init,
        n_iter=n_iter,
        n_candidates=n_candidates,
        seeds=seeds,
        phi_in_pi=phi_in_pi,
        xi=xi,
        note="Kernel sweep with 5 seeds; convergence plots are mean ± std.",
    )

    results: Dict[str, Dict[str, object]] = {}

    for kname, kobj in kernels:
        traces: List[np.ndarray] = []
        best_xs: List[np.ndarray] = []
        best_fs: List[float] = []
        for sd in seeds:
            cache: dict[tuple[float, ...], float] = {}

            def objective(x: np.ndarray) -> float:
                x = np.asarray(x, dtype=float).reshape(-1)
                key = tuple(round(float(v), 10) for v in x)
                if key in cache:
                    return cache[key]
                try:
                    th, ph, ps = (float(v) for v in x)
                    V = analyzer.parameterised_unitary(th, ph, ps)
                    choi_input = analyzer.choi_state_unitary(V)
                    choi_output = analyzer.link_product(choi_superchannel, choi_input)
                    _, F_U = analyzer.closest_unitary_channel(choi_output)
                    val = float(np.real(F_U))
                except Exception:
                    val = 0.0
                val = float(np.clip(val, 0.0, 1.0))
                cache[key] = val
                return val

            _, _, best_hist, best_x, best_f = _run_bo_single_seed(
                objective=objective,
                bounds=bounds,
                kernel_name=kname,
                kernel_obj=kobj,
                n_init=n_init,
                n_iter=n_iter,
                n_candidates=n_candidates,
                seed=sd,
                xi=xi,
                verbose=True,
                do_local_refine=do_local_refine,
                local_refine_steps=local_refine_steps,
                local_refine_sigma=local_refine_sigma,
            )
            traces.append(best_hist)
            best_xs.append(best_x)
            best_fs.append(float(best_f))

        Tlen = n_init + n_iter + (local_refine_steps if do_local_refine else 0)
        mat = np.vstack([tr[:Tlen] for tr in traces])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        csv_path = os.path.join(mode_dir, f"single_slot_{sanitize_filename(kname)}_seeds.csv")
        header = "eval_idx," + ",".join([f"seed_{sd}" for sd in seeds]) + ",mean,std"
        eval_idx = np.arange(1, Tlen + 1, dtype=int)
        out_mat = np.column_stack([eval_idx, mat.T, mean, std])
        np.savetxt(csv_path, out_mat, delimiter=",", header=header, comments="")

        import matplotlib.pyplot as plt

        png_path = os.path.join(mode_dir, f"single_slot_{sanitize_filename(kname)}_convergence.png")
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(eval_idx, mean, linewidth=2.4, label=f"{kname} mean")
        plt.fill_between(eval_idx, mean - std, mean + std, alpha=0.20, label="±1 std")
        plt.axvline(x=n_init, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="end of init")
        plt.axvline(x=n_init + n_iter, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="end of BO")
        plt.title(f"SingleSlot BO convergence | {kname} | {len(seeds)} seeds")
        plt.xlabel("Evaluation")
        plt.ylabel("Best-so-far Fidelity $F_U$")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close()

        _append_best_solutions(
            mode_dir,
            kernel_name=kname,
            seeds=seeds,
            best_fs=best_fs,
            best_xs=best_xs,
        )

        results[kname] = {
            "seeds": seeds,
            "best_histories": traces,
            "best_fs": best_fs,
            "best_xs": best_xs,
            "mean": mean,
            "std": std,
            "csv_path": csv_path,
            "png_path": png_path,
            "mode_dir": mode_dir,
            "run_root": run_root,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BO/GP benchmarking for SingleSlot (3 vars) and DoubleSlot (6 vars).")
    parser.add_argument("--mode", choices=["single", "double"], default="double", help="Which problem to run.")
    parser.add_argument(
        "--double_reduced3",
        action="store_true",
        help="(double mode only) Force theta1=0, theta2=0, psi2=0; optimize reduced 3-variable problem (phi1,psi1,phi2).",
    )
    # Backward-compatible alias (old name). Same behavior as --double_reduced3.
    parser.add_argument(
        "--double_reduced4",
        dest="double_reduced3",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.mode == "single":
        results = run_single_slot_kernel_seed_sweep(
            g=(0.8, 1.6),
            r=0.3,
            t=10/3,
            n_init=20,
            n_iter=100,
            n_candidates=8000,
            seeds=[1, 2, 3, 4, 5],
            phi_in_pi=True,
            xi=0.01,
            do_local_refine=False,
            local_refine_steps=0,
            local_refine_sigma=0.15,
        )
        prefix = "single_slot"
    else:
        results = run_double_slot_kernel_seed_sweep(
            g=(0.8, 0.8, 0.8),
            r=0.3,
            t=10/3,
            n_init=20,
            n_iter=100,
            n_candidates=8000,
            seeds=[1, 2, 3, 4, 5],
            phi_in_pi=True,
            xi=0.01,
            do_local_refine=False,
            local_refine_steps=0,
            local_refine_sigma=0.15,
            reduced3_fix_th1_th2_ps2_zero=args.double_reduced3,
        )
        prefix = "double_slot"

    # Print artifact paths for convenience (also prints the per-run folders)
    any_payload = next(iter(results.values()))
    print(f"\n=== Run folder ===")
    print(f"run_root: {any_payload['run_root']}")
    print(f"mode_dir: {any_payload['mode_dir']}")

    print(f"\n=== Saved convergence plots (mean ± std) | mode={prefix} ===")
    for kname, payload in results.items():
        print(f"{kname}:")
        print(f"  png: {payload['png_path']}")
        print(f"  csv: {payload['csv_path']}")


if __name__ == "__main__":
    main()


