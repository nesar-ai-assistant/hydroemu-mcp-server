"""MCP tool functions for cosmological hydrodynamic simulation emulators.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: predict_observable writes one
CSV per prediction, plot_prediction and plot_observable_comparison read them
back. Only small metadata ever passes through the LLM context.

Powered by the ``cosmohydro_emu`` package:
https://github.com/nesar/cosmohydro_emu
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import numpy as np
from pydantic import Field, validate_call

# All 14 valid observable names as a Literal type
ObservableName = Literal[
    "GSMF", "HMF", "fGas", "Pk-ratio", "CSFR",
    "CGD", "CGED", "CPP", "CTP", "CEP", "CEEP", "CMP", "CYP",
    "Pk_GO",
]

__all__ = [
    "list_observables",
    "describe_parameters",
    "predict_observable",
    "plot_prediction",
    "plot_observable_comparison",
]


@dataclass
class ArtifactResult:
    """Uniform result contract returned by every tool."""

    status: str
    message: str
    files: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool 1: list_observables
# ---------------------------------------------------------------------------


@validate_call
def list_observables() -> ArtifactResult:
    """List all available observables from the HACC hydrodynamic simulation emulators.

    Use this tool first to discover what summary statistics can be predicted,
    their physical descriptions, available redshift ranges, parameter counts,
    and categories. Each observable has a short key (e.g. "GSMF") used by
    predict_observable and plot_prediction.

    Available observables include galaxy/halo statistics (GSMF, HMF, fGas),
    matter power spectrum diagnostics (Pk-ratio, Pk_GO, CSFR), and
    cluster thermodynamic profiles (CGD, CGED, CPP, CTP, CEP, CEEP, CMP, CYP).

    Most statistics take 7 input parameters (5 subgrid + 2 cosmology). The
    gravity-only statistic Pk_GO takes only 2 cosmology parameters.
    """
    from cosmohydro_emu import get_statistic_info

    info = get_statistic_info()
    catalog = {}
    for name, meta in info.items():
        catalog[name] = {
            "description": meta["title"],
            "category": meta["category"],
            "n_params": meta["n_params"],
            "n_x_bins": meta["n_x"],
            "x_description": meta["x_description"],
            "redshifts": np.round(meta["redshifts"], 4).tolist(),
            "z_range": [
                round(float(meta["redshifts"].min()), 4),
                round(float(meta["redshifts"].max()), 4),
            ],
        }
    return ArtifactResult(
        status="success",
        message=f"{len(catalog)} observables available: {', '.join(catalog)}.",
        files=[],
        metadata={"observables": catalog},
    )


# ---------------------------------------------------------------------------
# Tool 2: describe_parameters
# ---------------------------------------------------------------------------


@validate_call
def describe_parameters(
    stat_name: Annotated[
        Optional[ObservableName],
        Field(
            default=None,
            description=(
                "Observable name to show parameters for. "
                "If omitted, returns all 7 parameters. "
                "For gravity-only stats like Pk_GO, returns only the 2 cosmology parameters."
            ),
        ),
    ] = None,
) -> ArtifactResult:
    """Describe the input parameter space of the HACC cosmological hydro emulators.

    Returns the subgrid parameters (kappa_w, e_w, M_seed, v_kin, epsilon_kin)
    and cosmology parameters (omega_m, sigma_8) with their physical
    descriptions, LaTeX symbols, design ranges, and scaling factors.

    Use these ranges when calling predict_observable — values outside the
    design will trigger extrapolation warnings.

    The 110-simulation Latin hypercube design covers 400 Mpc/h boxes with
    HACC's CRK-HACC hydrodynamics code varying AGN feedback, kinetic
    feedback, and cosmological parameters simultaneously.
    """
    from cosmohydro_emu import get_parameter_info

    pinfo = get_parameter_info(stat_name)
    params = {}
    for name in pinfo["names"]:
        lo, hi = pinfo["ranges"][name]
        params[name] = {
            "description": pinfo["descriptions"][name],
            "min": lo,
            "max": hi,
        }
        if name in pinfo.get("scales", {}):
            params[name]["scale"] = pinfo["scales"][name]

    label = stat_name or "all statistics"
    return ArtifactResult(
        status="success",
        message=(
            f"{len(params)}-parameter design space for {label}: "
            f"{', '.join(pinfo['names'])}."
        ),
        files=[],
        metadata={
            "parameters": params,
            "parameter_order": pinfo["names"],
            "latex_names": pinfo["latex_names"],
            "num_simulations": 110,
            "box_size": "400 Mpc/h",
        },
    )


# ---------------------------------------------------------------------------
# Tool 3: predict_observable
# ---------------------------------------------------------------------------


@validate_call
def predict_observable(
    stat_name: Annotated[
        ObservableName,
        Field(description="Which summary statistic to predict (e.g. 'GSMF', 'Pk_GO')."),
    ],
    omega_m: Annotated[
        float,
        Field(ge=0.12, le=0.155, description="Physical matter density omega_m = Omega_m h^2"),
    ],
    sigma_8: Annotated[
        float,
        Field(ge=0.7, le=0.9, description="Amplitude of matter fluctuations sigma_8"),
    ],
    kappa_w: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=2.0,
            le=4.0,
            description="AGN wind coupling efficiency (not needed for Pk_GO)",
        ),
    ] = None,
    e_w: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.2,
            le=1.0,
            description="AGN energy efficiency (not needed for Pk_GO)",
        ),
    ] = None,
    m_seed: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.6,
            le=2.0,
            description="Black hole seed mass in units of 10^6 M_sun (not needed for Pk_GO)",
        ),
    ] = None,
    v_kin: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.1,
            le=1.2,
            description="Kinetic feedback velocity in units of 10^4 km/s (not needed for Pk_GO)",
        ),
    ] = None,
    eps_kin: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.02,
            le=1.2,
            description="Kinetic feedback efficiency in units of 10^1 (not needed for Pk_GO)",
        ),
    ] = None,
    z: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description=(
                "Target redshift. Must be within the observable's trained range. "
                "Values between trained snapshots are linearly interpolated."
            ),
        ),
    ] = 0.0,
    output_dir: Annotated[
        str,
        Field(min_length=1, description="Directory where the output CSV is written"),
    ] = "/tmp/hydroemu",
) -> ArtifactResult:
    """Predict a summary statistic at a given redshift for specified parameters.

    Use this tool to generate emulator predictions for any of the 14 available
    observables. The prediction uses pre-trained SEPIA Gaussian Process models
    from the 110-simulation HACC Latin hypercube design, provided by the
    cosmohydro_emu package.

    For most statistics (GSMF, HMF, fGas, Pk-ratio, CSFR, and all cluster
    profiles), all 7 parameters are required. For the gravity-only statistic
    Pk_GO, only omega_m and sigma_8 are needed.

    The output CSV has columns: x (observable-specific independent variable),
    y_mean (GP posterior mean), y_std (GP posterior standard deviation). Pass
    the returned file path to plot_prediction or plot_observable_comparison —
    never copy raw numbers.
    """
    from cosmohydro_emu import load_emulator, get_plot_info

    # Build the parameter vector
    if stat_name == "Pk_GO":
        params = [omega_m, sigma_8]
    else:
        # All 5 subgrid params are required for non-gravity-only stats
        missing = []
        if kappa_w is None:
            missing.append("kappa_w")
        if e_w is None:
            missing.append("e_w")
        if m_seed is None:
            missing.append("m_seed")
        if v_kin is None:
            missing.append("v_kin")
        if eps_kin is None:
            missing.append("eps_kin")
        if missing:
            raise ValueError(
                f"{stat_name} requires 7 parameters but these subgrid parameters "
                f"are missing: {', '.join(missing)}. "
                f"Only Pk_GO uses 2 cosmology parameters alone."
            )
        params = [kappa_w, e_w, m_seed, v_kin, eps_kin, omega_m, sigma_8]

    # Load and predict
    emu = load_emulator(stat_name)
    y_mean, y_std = emu.predict(params, z=z)
    x_vals = emu.x_grid
    plot_info = get_plot_info(stat_name)

    # Write CSV
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{stat_name}_z{z:.4f}_prediction.csv"

    param_dict = {"omega_m": omega_m, "sigma_8": sigma_8}
    if stat_name != "Pk_GO":
        param_dict = {
            "kappa_w": kappa_w, "e_w": e_w, "m_seed": m_seed,
            "v_kin": v_kin, "eps_kin": eps_kin, **param_dict,
        }
    param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
    label = f"{stat_name} z={z:.4f} ({param_str})"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# label: {label}\n")
        f.write(f"# observable: {stat_name}\n")
        f.write(f"# redshift: {z}\n")
        f.write(f"# x_label: {plot_info['xlabel']}\n")
        f.write(f"# y_label: {plot_info['ylabel']}\n")
        writer = csv.writer(f)
        writer.writerow(["x", "y_mean", "y_std"])
        for xi, yi, si in zip(x_vals.flat, y_mean.flat, y_std.flat):
            writer.writerow([xi, yi, si])

    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Predicted {stat_name} at z={z:.4f} "
            f"({len(x_vals.flat)} bins)."
        ),
        metadata={
            "observable": stat_name,
            "z": z,
            "n_bins": int(len(x_vals.flat)),
            "parameters": param_dict,
        },
    )


# ---------------------------------------------------------------------------
# Tool 4: plot_prediction
# ---------------------------------------------------------------------------


@validate_call
def plot_prediction(
    stat_name: Annotated[
        ObservableName,
        Field(description="Which summary statistic to predict and plot."),
    ],
    omega_m: Annotated[
        float,
        Field(ge=0.12, le=0.155, description="Physical matter density omega_m"),
    ],
    sigma_8: Annotated[
        float,
        Field(ge=0.7, le=0.9, description="Amplitude of matter fluctuations sigma_8"),
    ],
    kappa_w: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=2.0,
            le=4.0,
            description="AGN wind coupling efficiency (not needed for Pk_GO)",
        ),
    ] = None,
    e_w: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.2,
            le=1.0,
            description="AGN energy efficiency (not needed for Pk_GO)",
        ),
    ] = None,
    m_seed: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.6,
            le=2.0,
            description="Black hole seed mass in units of 10^6 M_sun (not needed for Pk_GO)",
        ),
    ] = None,
    v_kin: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.1,
            le=1.2,
            description="Kinetic feedback velocity in units of 10^4 km/s (not needed for Pk_GO)",
        ),
    ] = None,
    eps_kin: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.02,
            le=1.2,
            description="Kinetic feedback efficiency in units of 10^1 (not needed for Pk_GO)",
        ),
    ] = None,
    z: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Target redshift for the prediction.",
        ),
    ] = 0.0,
    output_dir: Annotated[
        str,
        Field(min_length=1, description="Directory where the PNG is written"),
    ] = "/tmp/hydroemu",
) -> ArtifactResult:
    """Generate a publication-quality plot of an emulator prediction with uncertainty band.

    Produces a single-panel figure showing the GP posterior mean with a 2-sigma
    shaded uncertainty band. Uses proper axis labels, scales, and title from
    the cosmohydro_emu metadata.

    For most statistics, all 7 parameters are required. For Pk_GO, only
    omega_m and sigma_8 are needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cosmohydro_emu import load_emulator, get_plot_info

    # Build the parameter vector
    if stat_name == "Pk_GO":
        params = [omega_m, sigma_8]
    else:
        missing = []
        if kappa_w is None:
            missing.append("kappa_w")
        if e_w is None:
            missing.append("e_w")
        if m_seed is None:
            missing.append("m_seed")
        if v_kin is None:
            missing.append("v_kin")
        if eps_kin is None:
            missing.append("eps_kin")
        if missing:
            raise ValueError(
                f"{stat_name} requires 7 parameters but these subgrid parameters "
                f"are missing: {', '.join(missing)}."
            )
        params = [kappa_w, e_w, m_seed, v_kin, eps_kin, omega_m, sigma_8]

    # Load, predict, plot
    emu = load_emulator(stat_name)
    y_mean, y_std = emu.predict(params, z=z)
    x_grid = emu.x_grid
    info = get_plot_info(stat_name)

    fig, ax = plt.subplots(figsize=(7, 5))
    line, = ax.plot(x_grid, y_mean, lw=2, color="C0", label=f"z = {z:.2f}")
    ax.fill_between(
        x_grid,
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        alpha=0.25,
        color=line.get_color(),
        lw=0,
        label=r"$\pm 2\sigma$",
    )
    ax.set_xscale(info["xscale"])
    ax.set_yscale(info["yscale"])
    ax.set_xlabel(info["xlabel"], fontsize=12)
    ax.set_ylabel(info["ylabel"], fontsize=12)
    ax.set_title(info["title"], fontsize=13)
    ax.legend(loc="best", fontsize=10)

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / f"{stat_name}_z{z:.4f}_prediction.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    param_dict = {"omega_m": omega_m, "sigma_8": sigma_8}
    if stat_name != "Pk_GO":
        param_dict = {
            "kappa_w": kappa_w, "e_w": e_w, "m_seed": m_seed,
            "v_kin": v_kin, "eps_kin": eps_kin, **param_dict,
        }

    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=f"Plotted {stat_name} prediction at z={z:.4f}.",
        metadata={
            "observable": stat_name,
            "z": z,
            "parameters": param_dict,
        },
    )


# ---------------------------------------------------------------------------
# Tool 5: plot_observable_comparison
# ---------------------------------------------------------------------------


@validate_call
def plot_observable_comparison(
    prediction_files: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=10,
            description=(
                "Paths of CSV files written by predict_observable"
            ),
        ),
    ],
    output_dir: Annotated[
        str,
        Field(min_length=1, description="Directory where the PNG is written"),
    ],
    stat_name: Annotated[
        ObservableName,
        Field(
            default="GSMF",
            description="Observable name for axis labels and scales.",
        ),
    ] = "GSMF",
    reference_index: Annotated[
        int,
        Field(ge=0, description="Which file is the ratio reference (0 = first)"),
    ] = 0,
) -> ArtifactResult:
    """Plot multiple emulator predictions overlaid for comparison.

    Use this tool after predict_observable. It reads the CSV files produced
    by that tool and draws a two-panel figure: the observable values in the
    top panel and the ratio of each curve to a reference curve in the bottom
    panel.

    This is useful for comparing predictions across different parameter
    choices or redshifts to understand sensitivity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cosmohydro_emu import get_plot_info

    if reference_index >= len(prediction_files):
        raise ValueError("reference_index is out of range for prediction_files.")

    info = get_plot_info(stat_name)

    curves = []
    for path_str in prediction_files:
        path = Path(path_str).expanduser().resolve()
        lines = path.read_text(encoding="utf-8").splitlines()
        # Extract label from header
        label = path.stem
        for line in lines:
            if line.startswith("# label:"):
                label = line.removeprefix("# label:").strip()
                break

        # Count header lines
        n_header = sum(1 for line in lines if line.startswith("#"))
        data = np.loadtxt(path, delimiter=",", skiprows=n_header + 1)
        x, y_mean = data[:, 0], data[:, 1]
        y_std = data[:, 2] if data.shape[1] > 2 else np.zeros_like(y_mean)
        curves.append({"label": label, "x": x, "y": y_mean, "y_std": y_std})

    ref = curves[reference_index]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )
    linestyles = ["-", "--", "-.", ":"]
    for i, c in enumerate(curves):
        ls = linestyles[i % len(linestyles)]
        ax1.plot(c["x"], c["y"], ls, color=f"C{i}", linewidth=1.8, label=c["label"])
        if np.any(c["y_std"] > 0):
            ax1.fill_between(
                c["x"], c["y"] - c["y_std"], c["y"] + c["y_std"],
                color=f"C{i}", alpha=0.15,
            )
    ax1.set_ylabel(info["ylabel"])
    ax1.set_title(f"{info['title']}: emulator predictions")
    ax1.legend(loc="best", fontsize="x-small")
    ax1.set_xscale(info["xscale"])
    ax1.set_yscale(info["yscale"])

    for i, c in enumerate(curves):
        if i == reference_index:
            continue
        # Interpolate reference to match curve x-values
        ref_y_interp = np.interp(c["x"], ref["x"], ref["y"])
        mask = ref_y_interp != 0
        ratio = np.ones_like(c["y"])
        ratio[mask] = c["y"][mask] / ref_y_interp[mask]
        ax2.plot(c["x"], ratio, linestyles[i % len(linestyles)],
                 color=f"C{i}", linewidth=1.8)
    ax2.axhline(1.0, color="black", linewidth=1)
    ax2.set_xlabel(info["xlabel"])
    ax2.set_ylabel("ratio to ref")
    ax2.set_ylim(0.5, 1.5)

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / f"{stat_name}_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=f"Plotted {len(curves)} {stat_name} predictions.",
        metadata={
            "observable": stat_name,
            "labels": [c["label"] for c in curves],
            "reference": ref["label"],
        },
    )
