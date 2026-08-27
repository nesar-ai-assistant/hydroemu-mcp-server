"""MCP tool functions for cosmological hydrodynamic simulation emulators.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: predict_observable writes one
CSV per prediction, plot_observable_comparison reads them back. Only small
metadata ever passes through the LLM context.
"""

import csv
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from .emulator import (
    OBSERVABLE_CATALOG,
    PARAMETER_SPACE,
    PARAM_NAMES_ORDERED,
)

# All valid observable names as a Literal type
ObservableName = Literal[
    "GSMF", "HMF", "fGas", "CGD", "CGED", "CPP", "CTP", "CEP", "CEEP", "CMP", "CYP"
]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


@validate_call
def list_observables() -> ArtifactResult:
    """List all available observables from the HACC hydrodynamic simulation emulators.

    Use this tool first to discover what summary statistics can be predicted,
    their physical descriptions, available redshift ranges, and the number of
    trained snapshot models. Each observable has a short key (e.g. "GSMF") used
    by predict_observable and predict_observable_redshift.

    Available observables include galaxy/halo statistics (GSMF, HMF, fGas) and
    cluster thermodynamic profiles (CGD, CGED, CPP, CTP, CEP, CEEP, CMP, CYP).
    """
    catalog = {}
    for name, info in OBSERVABLE_CATALOG.items():
        catalog[name] = {
            "description": info["description"],
            "x_label": info["x_label"],
            "y_label": info["y_label"],
            "num_snapshots": info["num_snapshots"],
            "z_range": info["z_range"],
            "available_redshifts": info["redshifts"],
        }
    return ArtifactResult(
        status="success",
        files=[],
        message=f"{len(catalog)} observables available: {', '.join(catalog)}.",
        metadata={"observables": catalog},
    )


@validate_call
def describe_parameters() -> ArtifactResult:
    """Describe the 7-parameter space of the HACC cosmological hydro emulators.

    Returns the 5 subgrid parameters (kappa_w, e_w, m_seed, v_kin, eps_kin)
    and 2 cosmology parameters (omega_m, sigma_8) with their physical
    descriptions, units, symbols, and design ranges. Use these ranges when
    calling predict_observable — values outside the design will be rejected.

    The 110-simulation Latin hypercube design covers 400 Mpc/h boxes with
    HACC's CRK-HACC hydrodynamics code varying AGN feedback, kinetic
    feedback, and cosmological parameters simultaneously.
    """
    params = {}
    for name in PARAM_NAMES_ORDERED:
        info = PARAMETER_SPACE[name]
        params[name] = {
            "description": info["description"],
            "symbol": info["symbol"],
            "units": info["units"],
            "min": info["range"][0],
            "max": info["range"][1],
        }
    return ArtifactResult(
        status="success",
        files=[],
        message=f"7-parameter design space: {', '.join(PARAM_NAMES_ORDERED)}.",
        metadata={
            "parameters": params,
            "parameter_order": PARAM_NAMES_ORDERED,
            "num_simulations": 110,
            "box_size": "400 Mpc/h",
        },
    )


@validate_call
def predict_observable(
    observable: ObservableName,
    kappa_w: Annotated[float, Field(ge=0.03, le=3.0,
        description="AGN wind coupling efficiency")],
    e_w: Annotated[float, Field(ge=0.001, le=0.1,
        description="AGN energy efficiency")],
    m_seed: Annotated[float, Field(ge=0.5, le=50.0,
        description="Black hole seed mass in units of 10^6 M_sun")],
    v_kin: Annotated[float, Field(ge=0.1, le=1.0,
        description="Kinetic feedback velocity in units of 10^4 km/s")],
    eps_kin: Annotated[float, Field(ge=0.1, le=1.0,
        description="Kinetic feedback efficiency in units of 10^1")],
    omega_m: Annotated[float, Field(ge=0.12, le=0.155,
        description="Matter density parameter omega_m")],
    sigma_8: Annotated[float, Field(ge=0.7, le=0.9,
        description="Amplitude of matter fluctuations sigma_8")],
    output_dir: Annotated[str, Field(min_length=1,
        description="Directory where the output CSV is written")],
) -> ArtifactResult:
    """Predict a single observable at z=0 for given cosmological and subgrid parameters.

    Use this tool to generate emulator predictions for any of the available
    observables at redshift zero. The prediction uses pre-trained SEPIA
    Gaussian Process models from the 110-simulation HACC Latin hypercube
    design.

    The output CSV has columns: x (observable-specific independent variable),
    y_mean (GP posterior mean), y_std (GP posterior standard deviation). Pass
    the returned file path to plot_observable_comparison — never copy raw
    numbers.

    Args:
        observable: Which summary statistic to predict (e.g. "GSMF", "CGD").
        kappa_w: AGN wind coupling [0.03, 3.0].
        e_w: AGN energy efficiency [0.001, 0.1].
        m_seed: BH seed mass / 10^6 M_sun [0.5, 50.0].
        v_kin: Kinetic velocity / 10^4 km/s [0.1, 1.0].
        eps_kin: Kinetic efficiency / 10^1 [0.1, 1.0].
        omega_m: Matter density parameter [0.12, 0.155].
        sigma_8: Fluctuation amplitude [0.7, 0.9].
        output_dir: Directory for the output CSV.
    """
    from .emulator import (
        predict,
        load_training_data,
        load_models_for_observable,
        OBSERVABLE_CATALOG,
    )

    catalog = OBSERVABLE_CATALOG[observable]
    input_params = np.array([kappa_w, e_w, m_seed, v_kin, eps_kin, omega_m, sigma_8])

    # Load bundled training data arrays
    td = load_training_data(observable)
    p_train = td["p_train"]
    y_vals = td["y_vals"]
    y_ind = td["y_ind"]
    z_index_range = td["z_index_range"]

    # Load all snapshot models (cached after first call)
    model_list, data_list = load_models_for_observable(
        observable,
        y_vals_all=y_vals,
        y_ind_all=y_ind,
        p_train_all=p_train,
        z_index_range=z_index_range,
    )

    # Use z=0 snapshot (last in z_index_range)
    z_index = len(model_list) - 1
    y_mean, y_std = predict(model_list[z_index], input_params, sepia_data=data_list[z_index])
    x_vals = y_ind

    # Write CSV
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{observable}_z0_prediction.csv"
    label = f"{observable} z=0 (κ_w={kappa_w}, e_w={e_w}, m_seed={m_seed}, v_kin={v_kin}, ε_kin={eps_kin}, ω_m={omega_m}, σ_8={sigma_8})"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# label: {label}\n")
        f.write(f"# observable: {observable}\n")
        f.write(f"# x_label: {catalog['x_label']}\n")
        f.write(f"# y_label: {catalog['y_label']}\n")
        writer = csv.writer(f)
        writer.writerow(["x", "y_mean", "y_std"])
        for xi, yi, si in zip(x_vals.flat, y_mean.flat, y_std.flat):
            writer.writerow([xi, yi, si])

    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=f"Predicted {observable} at z=0 ({len(x_vals.flat)} bins).",
        metadata={
            "observable": observable,
            "z": 0.0,
            "parameters": {
                "kappa_w": kappa_w,
                "e_w": e_w,
                "m_seed": m_seed,
                "v_kin": v_kin,
                "eps_kin": eps_kin,
                "omega_m": omega_m,
                "sigma_8": sigma_8,
            },
        },
    )


@validate_call
def predict_observable_redshift(
    observable: ObservableName,
    kappa_w: Annotated[float, Field(ge=0.03, le=3.0,
        description="AGN wind coupling efficiency")],
    e_w: Annotated[float, Field(ge=0.001, le=0.1,
        description="AGN energy efficiency")],
    m_seed: Annotated[float, Field(ge=0.5, le=50.0,
        description="Black hole seed mass in units of 10^6 M_sun")],
    v_kin: Annotated[float, Field(ge=0.1, le=1.0,
        description="Kinetic feedback velocity in units of 10^4 km/s")],
    eps_kin: Annotated[float, Field(ge=0.1, le=1.0,
        description="Kinetic feedback efficiency in units of 10^1")],
    omega_m: Annotated[float, Field(ge=0.12, le=0.155,
        description="Matter density parameter omega_m")],
    sigma_8: Annotated[float, Field(ge=0.7, le=0.9,
        description="Amplitude of matter fluctuations sigma_8")],
    redshift: Annotated[float, Field(ge=0.0, le=2.0,
        description="Target redshift for interpolation")],
    output_dir: Annotated[str, Field(min_length=1,
        description="Directory where the output CSV is written")],
) -> ArtifactResult:
    """Predict a single observable at an arbitrary redshift using interpolation.

    Like predict_observable but for any redshift within the trained range.
    The emulator linearly interpolates between the two nearest snapshot
    models bracketing the requested redshift.

    Check list_observables() for the redshift range of each observable:
    GSMF and HMF cover z=0–2, fGas covers z=0–1.0, and cluster profiles
    cover z=0–0.5.

    Args:
        observable: Which summary statistic to predict (e.g. "GSMF", "CGD").
        kappa_w: AGN wind coupling [0.03, 3.0].
        e_w: AGN energy efficiency [0.001, 0.1].
        m_seed: BH seed mass / 10^6 M_sun [0.5, 50.0].
        v_kin: Kinetic velocity / 10^4 km/s [0.1, 1.0].
        eps_kin: Kinetic efficiency / 10^1 [0.1, 1.0].
        omega_m: Matter density parameter [0.12, 0.155].
        sigma_8: Fluctuation amplitude [0.7, 0.9].
        redshift: Target redshift (linearly interpolated between snapshots).
        output_dir: Directory for the output CSV.
    """
    from .emulator import (
        predict_at_redshift,
        load_training_data,
        load_models_for_observable,
        OBSERVABLE_CATALOG,
        get_snapshot_redshifts,
    )

    catalog = OBSERVABLE_CATALOG[observable]
    z_min, z_max = catalog["z_range"]
    if redshift < z_min or redshift > z_max:
        raise ValueError(
            f"Redshift {redshift} is outside the trained range "
            f"[{z_min}, {z_max}] for {observable}. "
            f"Check list_observables() for valid ranges."
        )

    input_params = np.array([kappa_w, e_w, m_seed, v_kin, eps_kin, omega_m, sigma_8])

    # Load bundled training data arrays
    td = load_training_data(observable)
    p_train = td["p_train"]
    y_vals = td["y_vals"]
    y_ind = td["y_ind"]
    z_index_range = td["z_index_range"]

    # Get snapshot redshifts for this observable
    z_all, _ = get_snapshot_redshifts(catalog["snapshot_ids"])

    # Load all snapshot models (cached after first call)
    model_list, data_list = load_models_for_observable(
        observable,
        y_vals_all=y_vals,
        y_ind_all=y_ind,
        p_train_all=p_train,
        z_index_range=z_index_range,
    )

    # Predict with redshift interpolation
    y_mean, y_std = predict_at_redshift(
        input_params, redshift, model_list, data_list, z_all
    )

    # x-values from the training data
    x_vals = y_ind

    # Write CSV
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{observable}_z{redshift:.4f}_prediction.csv"
    label = (
        f"{observable} z={redshift:.4f} "
        f"(κ_w={kappa_w}, e_w={e_w}, m_seed={m_seed}, "
        f"v_kin={v_kin}, ε_kin={eps_kin}, ω_m={omega_m}, σ_8={sigma_8})"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# label: {label}\n")
        f.write(f"# observable: {observable}\n")
        f.write(f"# redshift: {redshift}\n")
        f.write(f"# x_label: {catalog['x_label']}\n")
        f.write(f"# y_label: {catalog['y_label']}\n")
        writer = csv.writer(f)
        writer.writerow(["x", "y_mean", "y_std"])
        for xi, yi, si in zip(x_vals.flat, y_mean.flat, y_std.flat):
            writer.writerow([xi, yi, si])

    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Predicted {observable} at z={redshift:.4f} "
            f"({len(x_vals.flat)} bins, interpolated)."
        ),
        metadata={
            "observable": observable,
            "z": redshift,
            "parameters": {
                "kappa_w": kappa_w,
                "e_w": e_w,
                "m_seed": m_seed,
                "v_kin": v_kin,
                "eps_kin": eps_kin,
                "omega_m": omega_m,
                "sigma_8": sigma_8,
            },
        },
    )


@validate_call
def plot_observable_comparison(
    prediction_files: Annotated[list[str], Field(min_length=1, max_length=10,
        description="Paths of CSV files written by predict_observable or predict_observable_redshift")],
    output_dir: Annotated[str, Field(min_length=1,
        description="Directory where the PNG is written")],
    observable: ObservableName = "GSMF",
    reference_index: Annotated[int, Field(ge=0,
        description="Which file is the ratio reference (0 = first)")] = 0,
) -> ArtifactResult:
    """Plot multiple emulator predictions overlaid for comparison.

    Use this tool after predict_observable or predict_observable_redshift.
    It reads the CSV files produced by those tools and draws a two-panel
    figure: the observable values in the top panel and the ratio of each
    curve to a reference curve in the bottom panel.

    This is useful for comparing predictions across different parameter
    choices or redshifts to understand sensitivity.

    Args:
        prediction_files: Paths of CSV files from predict_observable/predict_observable_redshift.
        output_dir: Directory where the PNG is written.
        observable: Observable name for axis labels (default: GSMF).
        reference_index: Which entry of prediction_files is the ratio reference
            (0 = the first file).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reference_index >= len(prediction_files):
        raise ValueError("reference_index is out of range for prediction_files.")

    catalog = OBSERVABLE_CATALOG.get(observable, {})
    x_label = catalog.get("x_label", "x")
    y_label = catalog.get("y_label", "y")
    title = catalog.get("description", observable)

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
    ax1.set_ylabel(y_label)
    ax1.set_title(f"{title}: emulator predictions")
    ax1.legend(loc="best", fontsize="x-small")

    # Use log scale for mass functions and gas fraction
    if observable in ("GSMF", "HMF", "fGas"):
        ax1.set_yscale("log")

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
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(f"ratio to ref")
    ax2.set_ylim(0.5, 1.5)

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / f"{observable}_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=f"Plotted {len(curves)} {observable} predictions.",
        metadata={
            "observable": observable,
            "labels": [c["label"] for c in curves],
            "reference": ref["label"],
        },
    )
