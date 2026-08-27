"""Core emulator wrapper around SEPIA GP models.

This module provides lazy-loading and prediction functions for pre-trained
SEPIA Gaussian Process emulators of HACC cosmological hydrodynamic
simulations. Models are loaded from pickle files in the ``models/`` directory
relative to the repository root.

All SEPIA imports are lazy (inside functions, not at module scope) so that
the tools package can be imported and tested without the full sepia stack.
"""

__all__ = [
    "REPO_ROOT",
    "MODELS_DIR",
    "OBSERVABLE_CATALOG",
    "PARAMETER_SPACE",
    "SNAPSHOT_IDS",
    "get_snapshot_redshifts",
    "load_training_data",
    "load_models_for_observable",
    "predict",
    "predict_at_redshift",
]

import contextlib
import io
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Monkey-patch SEPIA for scipy >= 1.14 compatibility
# scipy.linalg.solve() removed the sym_pos keyword; SEPIA still uses it.
# We patch scipy.linalg.solve before SEPIA imports it.
# ---------------------------------------------------------------------------
import scipy.linalg as _sla
_orig_solve = _sla.solve


def _patched_solve(*args, **kwargs):
    if "sym_pos" in kwargs:
        kwargs.pop("sym_pos")
        kwargs.setdefault("assume_a", "pos")
    return _orig_solve(*args, **kwargs)


_sla.solve = _patched_solve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# Snapshot / redshift mapping (from CosmoHydro snapshot_utils)
# ---------------------------------------------------------------------------

SNAPSHOT_IDS = [205, 224, 247, 275, 310, 355, 415, 479, 498, 567, 624]


def _scale_factor_from_snapshot(snapshot_number: int,
                                z_initial: float = 200.0,
                                n_snaps: int = 625) -> float:
    a_min = 1.0 / (1.0 + z_initial)
    return a_min + (1.0 - a_min) * ((snapshot_number + 1.0) / n_snaps)


def get_snapshot_redshifts(
    snapshot_ids: Optional[list[int]] = None,
    z_initial: float = 200.0,
    n_snaps: int = 625,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute redshifts and scale factors for a list of snapshot IDs."""
    if snapshot_ids is None:
        snapshot_ids = SNAPSHOT_IDS
    scale_factors = np.array([
        _scale_factor_from_snapshot(s, z_initial, n_snaps)
        for s in snapshot_ids
    ])
    redshifts = 1.0 / scale_factors - 1.0
    return redshifts, scale_factors


# Pre-compute the standard 11-snapshot redshifts
_REDSHIFTS_11, _SCALE_FACTORS_11 = get_snapshot_redshifts()

# ---------------------------------------------------------------------------
# Observable catalog
# ---------------------------------------------------------------------------

OBSERVABLE_CATALOG = {
    "GSMF": {
        "description": "Galaxy Stellar Mass Function",
        "x_label": "log10(M_stars / M_sun)",
        "y_label": "dn/dlog10(M_stars) [(h^-1 Mpc)^-3]",
        "num_snapshots": 11,
        "snapshot_ids": SNAPSHOT_IDS,
        "z_range": [0.0, 2.0],
        "redshifts": _REDSHIFTS_11.round(4).tolist(),
        "model_subdir": "GSMF_multiz",
    },
    "HMF": {
        "description": "Halo Mass Function",
        "x_label": "M_halo [M_sun]",
        "y_label": "dn/dlog10(M) [(Mpc/h)^-3]",
        "num_snapshots": 11,
        "snapshot_ids": SNAPSHOT_IDS,
        "z_range": [0.0, 2.0],
        "redshifts": _REDSHIFTS_11.round(4).tolist(),
        "model_subdir": "HMF_multiz",
    },
    "fGas": {
        "description": "Cluster Gas Fraction",
        "x_label": "M_500c / (h^-1 M_sun)",
        "y_label": "M_gas / M_500c",
        "num_snapshots": 7,
        "snapshot_ids": SNAPSHOT_IDS[4:],  # last 7 snapshots (z<=1.0)
        "z_range": [0.0, 1.0],
        "redshifts": _REDSHIFTS_11[4:].round(4).tolist(),
        "model_subdir": "fGas_multiz",
    },
    "CGD": {
        "description": "Cluster Gas Density Profile",
        "x_label": "r / R_500c",
        "y_label": "rho_gas / rho_crit",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],  # last 5 snapshots (z<=0.5)
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CGD_multiz",
    },
    "CGED": {
        "description": "Cluster Gas Electron Density Profile",
        "x_label": "r / R_500c",
        "y_label": "n_e [cm^-3]",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CGED_multiz",
    },
    "CPP": {
        "description": "Cluster Gas Pressure Profile",
        "x_label": "r / R_500c",
        "y_label": "P / P_500",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CPP_multiz",
    },
    "CTP": {
        "description": "Cluster Gas Temperature Profile",
        "x_label": "r / R_500c",
        "y_label": "T / T_500",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CTP_multiz",
    },
    "CEP": {
        "description": "Cluster Gas Entropy Profile",
        "x_label": "r / R_500c",
        "y_label": "K / K_500",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CEP_multiz",
    },
    "CEEP": {
        "description": "Cluster Electron Entropy Profile",
        "x_label": "r / R_500c",
        "y_label": "K / K_500",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CEEP_multiz",
    },
    "CMP": {
        "description": "Cluster Gas Metallicity Profile",
        "x_label": "r / R_500c",
        "y_label": "Z / Z_sun",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CMP_multiz",
    },
    "CYP": {
        "description": "Cluster Compton-y (tSZ) Profile",
        "x_label": "r / R_500c",
        "y_label": "Y_SZ",
        "num_snapshots": 5,
        "snapshot_ids": SNAPSHOT_IDS[6:],
        "z_range": [0.0, 0.5],
        "redshifts": _REDSHIFTS_11[6:].round(4).tolist(),
        "model_subdir": "CYP_multiz",
    },
}

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

PARAMETER_SPACE = {
    "kappa_w": {
        "description": "AGN wind coupling efficiency",
        "symbol": "κ_w",
        "units": "dimensionless",
        "range": [0.03, 3.0],
        "scale": 1.0,
    },
    "e_w": {
        "description": "AGN energy efficiency",
        "symbol": "e_w",
        "units": "dimensionless",
        "range": [0.001, 0.1],
        "scale": 1.0,
    },
    "m_seed": {
        "description": "Black hole seed mass",
        "symbol": "M_seed",
        "units": "10^6 M_sun",
        "range": [0.5, 50.0],
        "scale": 1e6,
    },
    "v_kin": {
        "description": "Kinetic feedback velocity",
        "symbol": "v_kin",
        "units": "10^4 km/s",
        "range": [0.1, 1.0],
        "scale": 1e4,
    },
    "eps_kin": {
        "description": "Kinetic feedback efficiency",
        "symbol": "ε_kin",
        "units": "10^1",
        "range": [0.1, 1.0],
        "scale": 1e1,
    },
    "omega_m": {
        "description": "Matter density parameter (omega_m = Omega_m * h^2)",
        "symbol": "ω_m",
        "units": "dimensionless",
        "range": [0.12, 0.155],
        "scale": 1.0,
    },
    "sigma_8": {
        "description": "Amplitude of matter fluctuations at 8 Mpc/h",
        "symbol": "σ_8",
        "units": "dimensionless",
        "range": [0.7, 0.9],
        "scale": 1.0,
    },
}

PARAM_NAMES_ORDERED = [
    "kappa_w", "e_w", "m_seed", "v_kin", "eps_kin", "omega_m", "sigma_8"
]

# ---------------------------------------------------------------------------
# Lazy-loaded model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[list, list]] = {}

# Training data cache (npz arrays)
_training_data_cache: dict[str, dict] = {}


def load_training_data(observable: str) -> dict:
    """Load bundled training data arrays from the models/ directory.

    Each observable has a ``training_data.npz`` saved alongside its SEPIA
    pickles.  The npz contains:

    - ``p_train``: (100, 7) parameter design matrix (training sims only)
    - ``y_vals``: (100, n_snapshots, n_bins) observable values per snapshot
    - ``y_ind``: (n_bins,) the x-axis grid (mass bins, radii, k, etc.)
    - ``z_index_range``: snapshot indices that have trained models
    - ``redshifts``: redshift for each snapshot
    - ``param_names``: parameter names in order

    NaN values in ``y_vals`` are filled with linear interpolation along the
    bin axis (reproducing the CosmoHydro preprocessing step).

    Returns a dict with all arrays.  Results are cached.
    """
    if observable in _training_data_cache:
        return _training_data_cache[observable]

    catalog = OBSERVABLE_CATALOG[observable]
    npz_path = MODELS_DIR / catalog["model_subdir"] / "training_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {npz_path}. "
            "Run the export_training_data.py script from CosmoHydro to generate it."
        )

    data = dict(np.load(npz_path, allow_pickle=False))

    # Fill NaN values in y_vals (cluster profiles may have NaN bins)
    y_vals = data["y_vals"]
    if np.isnan(y_vals).any():
        y_vals = _fill_nan_interpolation(y_vals)
        data["y_vals"] = y_vals

    _training_data_cache[observable] = data
    return data


def _fill_nan_interpolation(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values in array by linear interpolation along the last axis.

    Reproduces CosmoHydro's fill_nan_with_interpolation for 2D and 3D arrays.
    For 3D arrays (n_sims, n_snapshots, n_bins), processes each (sim, snap) slice.
    """
    from scipy.interpolate import interp1d

    result = arr.copy()
    if result.ndim == 2:
        # (n_sims, n_bins)
        for i in range(result.shape[0]):
            row = result[i]
            mask = ~np.isnan(row)
            if mask.sum() >= 2 and (~mask).any():
                x = np.arange(len(row))
                f = interp1d(x[mask], row[mask], kind="linear",
                             fill_value="extrapolate", bounds_error=False)
                result[i] = f(x)
    elif result.ndim == 3:
        # (n_sims, n_snapshots, n_bins)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                row = result[i, j]
                mask = ~np.isnan(row)
                if mask.sum() >= 2 and (~mask).any():
                    x = np.arange(len(row))
                    f = interp1d(x[mask], row[mask], kind="linear",
                                 fill_value="extrapolate", bounds_error=False)
                    result[i, j] = f(x)
    return result


def _pu_from_saved_model(model_filename: str) -> Optional[int]:
    """Inspect a saved SEPIA model pickle and return the number of PCA basis
    components (pu) used at training. Returns None if not detectable."""
    path = model_filename if model_filename.endswith(".pkl") else model_filename + ".pkl"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception:
        return None
    samples = blob.get("samples") if isinstance(blob, dict) else None
    betaU = samples.get("betaU") if isinstance(samples, dict) else None
    if betaU is None or getattr(betaU, "ndim", 0) < 3:
        return None
    return int(betaU.shape[2])


def _load_model_autosync(model_filename: str, sepia_data, exp_variance: float = 0.95):
    """Load a trained SEPIA model with the PCA basis size auto-synced to the
    saved pickle's betaU shape."""
    from sepia.SepiaModel import SepiaModel  # noqa: F811 — lazy import

    pu = _pu_from_saved_model(model_filename)
    n_pc = pu if pu is not None else exp_variance

    with contextlib.redirect_stdout(io.StringIO()):
        # PCA transform
        sepia_data.transform_xt()
        sepia_data.standardize_y()
        sepia_data.create_K_basis(n_pc=n_pc)
        sepia_model = SepiaModel(sepia_data)
        # Restore trained hyperparameters
        sepia_model.restore_model_info(model_filename)

    return sepia_model


def load_models_for_observable(
    observable: str,
    *,
    y_vals_all: np.ndarray,
    y_ind_all: np.ndarray,
    p_train_all: np.ndarray,
    z_index_range: Optional[np.ndarray] = None,
    exp_variance: float = 0.95,
) -> tuple[list, list]:
    """Load all snapshot models for an observable.

    Returns (model_list, data_list) — one SepiaModel and SepiaData per
    snapshot index in z_index_range.

    This is the multi-snapshot ``load_model_multiple`` pattern from
    CosmoHydro's emu.py, adapted for the MCP server layout.

    Parameters
    ----------
    z_index_range : array-like, optional
        Global snapshot indices for which trained models exist.  When None,
        defaults to ``range(num_snapshots)`` (works for GSMF/HMF that have
        sequential indices 0..10).  For observables like fGas/CGD that only
        have models at a subset of snapshots (e.g., [4,5,6,...,10]), the
        caller **must** pass this from ``training_data.npz['z_index_range']``.
    """
    if observable in _model_cache:
        return _model_cache[observable]

    from sepia.SepiaData import SepiaData  # noqa — lazy import

    catalog = OBSERVABLE_CATALOG[observable]
    model_dir = MODELS_DIR / catalog["model_subdir"]

    if z_index_range is None:
        z_index_range = np.arange(catalog["num_snapshots"])

    model_list = []
    data_list = []

    for local_idx, z_index in enumerate(z_index_range):
        # y_vals_all may have shape (n_sims, n_all_snapshots, n_bins) with
        # the snapshot axis matching global snapshot indices, OR it may have
        # shape (n_sims, len(z_index_range), n_bins) already sliced.
        # Detect which layout we have by comparing axis-1 size.
        if y_vals_all.shape[1] > len(z_index_range):
            # Global layout: index by z_index (the global snapshot number)
            y_snap = y_vals_all[:, z_index, :]
        else:
            # Already sliced to match z_index_range
            y_snap = y_vals_all[:, local_idx, :]
        sepia_data = SepiaData(
            t_sim=p_train_all,
            y_sim=y_snap,
            y_ind_sim=y_ind_all,
        )
        model_filename = str(
            model_dir / f"multivariate_model_z_index{z_index}"
        )
        sepia_model = _load_model_autosync(
            model_filename, sepia_data, exp_variance=exp_variance
        )
        model_list.append(sepia_model)
        data_list.append(sepia_data)

    _model_cache[observable] = (model_list, data_list)
    return model_list, data_list


def predict(
    sepia_model,
    input_params: np.ndarray,
    sepia_data=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical GP predictor: returns (mean, std) on the original y-scale.

    Uses the latent GP posterior (mu, Sigma) and projects through the K basis:
        y_mu  = K^T mu
        y_std = sqrt(diag(K^T Sigma K))
    then undoes the standardization.
    """
    from sepia.SepiaPredict import SepiaEmulatorPrediction  # noqa — lazy

    if input_params.ndim == 1:
        input_params = np.expand_dims(input_params, 0)

    if sepia_data is None:
        sepia_data = sepia_model.data

    K = sepia_data.sim_data.K
    y_sd = sepia_data.sim_data.orig_y_sd
    y_mean = sepia_data.sim_data.orig_y_mean

    pred_samples = sepia_model.get_samples(numsamples=1)
    K_T = K.T

    means, stds = [], []
    for param in input_params:
        pred = SepiaEmulatorPrediction(
            t_pred=param[None, :],
            samples=pred_samples,
            model=sepia_model,
            storeMuSigma=True,
        )
        mu = pred.mu[0]
        Sigma = pred.sigma[0]

        y_mu = K_T @ mu
        y_cov = K_T @ Sigma @ K
        y_std_val = np.sqrt(np.clip(np.diag(y_cov), 0, None))

        y_mu = y_sd * y_mu + y_mean
        y_std_val = y_sd * y_std_val

        means.append(y_mu)
        stds.append(y_std_val)

    return np.stack(means, axis=1), np.stack(stds, axis=1)


def predict_at_redshift(
    input_params: np.ndarray,
    redshift: float,
    sepia_model_list: list,
    sepia_data_list: list,
    z_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the emulator across z snapshots.

    Returns (mean, std) on the original y-scale.
    """
    snap_idx_nearest = int(np.abs(z_all - redshift).argmin())
    if redshift > z_all[snap_idx_nearest]:
        snap_ID_z1 = snap_idx_nearest - 1
    else:
        snap_ID_z1 = snap_idx_nearest
    snap_ID_z2 = snap_ID_z1 + 1

    # Clamp to valid range
    snap_ID_z1 = max(0, min(snap_ID_z1, len(z_all) - 2))
    snap_ID_z2 = snap_ID_z1 + 1

    z1 = z_all[snap_ID_z1]
    z2 = z_all[snap_ID_z2]

    sd1 = sepia_data_list[snap_ID_z1] if sepia_data_list is not None else None
    sd2 = sepia_data_list[snap_ID_z2] if sepia_data_list is not None else None

    mean_z1, std_z1 = predict(sepia_model_list[snap_ID_z1], input_params, sepia_data=sd1)
    mean_z2, std_z2 = predict(sepia_model_list[snap_ID_z2], input_params, sepia_data=sd2)

    # Linear interpolation
    if abs(z1 - z2) < 1e-12:
        return mean_z1, std_z1
    frac = (redshift - z2) / (z1 - z2)
    mean_interp = mean_z2 + (mean_z1 - mean_z2) * frac
    std_interp = std_z2 + (std_z1 - std_z2) * frac

    return mean_interp, std_interp
