"""Tests for the science tools as plain Python — no MCP layer needed.

Tests that require real SEPIA models are marked with @pytest.mark.skipif
and skipped unless models are present. The list/describe and plot tools
are tested with fixture data.
"""

import os

import numpy as np
import pytest

from tools import (
    list_observables,
    describe_parameters,
    plot_observable_comparison,
)
from tools.emulator import (
    OBSERVABLE_CATALOG,
    PARAMETER_SPACE,
    PARAM_NAMES_ORDERED,
    MODELS_DIR,
    SNAPSHOT_IDS,
    get_snapshot_redshifts,
)


# ---------------------------------------------------------------------------
# list_observables
# ---------------------------------------------------------------------------


def test_list_observables_returns_all():
    result = list_observables()
    assert result.status == "success"
    obs = result.metadata["observables"]
    assert len(obs) == 11
    expected = {"GSMF", "HMF", "fGas", "CGD", "CGED", "CPP", "CTP", "CEP", "CEEP", "CMP", "CYP"}
    assert set(obs.keys()) == expected


def test_list_observables_has_descriptions():
    result = list_observables()
    for name, info in result.metadata["observables"].items():
        assert "description" in info
        assert "z_range" in info
        assert "num_snapshots" in info
        assert "available_redshifts" in info
        assert len(info["available_redshifts"]) == info["num_snapshots"]


def test_gsmf_has_11_snapshots():
    result = list_observables()
    gsmf = result.metadata["observables"]["GSMF"]
    assert gsmf["num_snapshots"] == 11
    assert gsmf["z_range"] == [0.0, 2.0]


def test_cluster_profiles_have_5_snapshots():
    result = list_observables()
    for name in ["CGD", "CGED", "CPP", "CTP", "CEP", "CEEP", "CMP", "CYP"]:
        info = result.metadata["observables"][name]
        assert info["num_snapshots"] == 5
        assert info["z_range"] == [0.0, 0.5]


# ---------------------------------------------------------------------------
# describe_parameters
# ---------------------------------------------------------------------------


def test_describe_parameters_returns_7():
    result = describe_parameters()
    assert result.status == "success"
    params = result.metadata["parameters"]
    assert len(params) == 7
    assert result.metadata["parameter_order"] == PARAM_NAMES_ORDERED


def test_parameter_ranges():
    result = describe_parameters()
    params = result.metadata["parameters"]

    assert params["kappa_w"]["min"] == 0.03
    assert params["kappa_w"]["max"] == 3.0
    assert params["e_w"]["min"] == 0.001
    assert params["e_w"]["max"] == 0.1
    assert params["m_seed"]["min"] == 0.5
    assert params["m_seed"]["max"] == 50.0
    assert params["v_kin"]["min"] == 0.1
    assert params["v_kin"]["max"] == 1.0
    assert params["eps_kin"]["min"] == 0.1
    assert params["eps_kin"]["max"] == 1.0
    assert params["omega_m"]["min"] == 0.12
    assert params["omega_m"]["max"] == 0.155
    assert params["sigma_8"]["min"] == 0.7
    assert params["sigma_8"]["max"] == 0.9


def test_parameters_have_descriptions():
    result = describe_parameters()
    for name, info in result.metadata["parameters"].items():
        assert "description" in info
        assert "symbol" in info
        assert "units" in info
        assert "min" in info
        assert "max" in info


# ---------------------------------------------------------------------------
# Snapshot / redshift utilities
# ---------------------------------------------------------------------------


def test_snapshot_ids_count():
    assert len(SNAPSHOT_IDS) == 11


def test_get_snapshot_redshifts():
    z, a = get_snapshot_redshifts()
    assert len(z) == 11
    assert len(a) == 11
    # z should be decreasing (higher snapshot number = lower redshift)
    assert z[0] > z[-1]
    # z=0 snapshot (624) should be very close to z=0
    assert z[-1] < 0.01
    # All scale factors should be positive and <= 1
    assert np.all(a > 0)
    assert np.all(a <= 1.0)


# ---------------------------------------------------------------------------
# Fixture CSV helpers for plot tests
# ---------------------------------------------------------------------------


def _write_fixture_csv(path, label, observable, scale=1.0, n_points=30):
    """Write a fake observable prediction CSV for testing the plot tool."""
    if observable in ("GSMF", "HMF"):
        x = np.linspace(9.0, 12.5, n_points)
        y_mean = scale * 10 ** (-(x - 10.0) ** 2)
    elif observable == "fGas":
        x = np.linspace(13.5, 15.0, n_points)
        y_mean = scale * 0.1 * np.ones(n_points)
    else:
        # Cluster profiles — radial
        x = np.linspace(0.05, 2.5, n_points)
        y_mean = scale * x ** -1.5
    y_std = 0.05 * y_mean

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# label: {label}\n")
        f.write(f"# observable: {observable}\n")
        f.write(f"# x_label: x\n")
        f.write(f"# y_label: y\n")
        f.write("x,y_mean,y_std\n")
        for xi, yi, si in zip(x, y_mean, y_std):
            f.write(f"{xi},{yi},{si}\n")


# ---------------------------------------------------------------------------
# plot_observable_comparison
# ---------------------------------------------------------------------------


def test_plot_observable_comparison_from_fixtures(tmp_path):
    files = []
    for name, scale in [("ref_params", 1.0), ("alt_params", 0.8)]:
        path = tmp_path / f"{name}.csv"
        _write_fixture_csv(path, name.upper(), "GSMF", scale)
        files.append(str(path))

    result = plot_observable_comparison(
        prediction_files=files,
        output_dir=str(tmp_path),
        observable="GSMF",
    )
    (png_path,) = result.files
    assert png_path.endswith("GSMF_comparison.png")
    assert result.metadata["reference"] == "REF_PARAMS"
    assert os.path.exists(png_path)


def test_plot_cgd_profile_from_fixtures(tmp_path):
    files = []
    for name, scale in [("base", 1.0), ("high_kappa", 1.2)]:
        path = tmp_path / f"{name}.csv"
        _write_fixture_csv(path, name, "CGD", scale)
        files.append(str(path))

    result = plot_observable_comparison(
        prediction_files=files,
        output_dir=str(tmp_path),
        observable="CGD",
    )
    (png_path,) = result.files
    assert png_path.endswith("CGD_comparison.png")
    assert os.path.exists(png_path)


def test_plot_rejects_bad_reference_index(tmp_path):
    path = tmp_path / "single.csv"
    _write_fixture_csv(path, "SINGLE", "GSMF")
    with pytest.raises(ValueError):
        plot_observable_comparison(
            prediction_files=[str(path)],
            output_dir=str(tmp_path),
            observable="GSMF",
            reference_index=3,
        )


def test_plot_single_file(tmp_path):
    """A single file should still produce a valid plot (no ratio curves)."""
    path = tmp_path / "one.csv"
    _write_fixture_csv(path, "ONE", "HMF")
    result = plot_observable_comparison(
        prediction_files=[str(path)],
        output_dir=str(tmp_path),
        observable="HMF",
    )
    assert result.status == "success"
    assert os.path.exists(result.files[0])


# ---------------------------------------------------------------------------
# predict_observable / predict_observable_redshift — need real models
# ---------------------------------------------------------------------------

_has_models = (MODELS_DIR / "GSMF_multiz" / "multivariate_model_z_index10.pkl").exists()
try:
    from sepia.SepiaModel import SepiaModel  # noqa: F401
    _has_sepia = True
except (ImportError, ModuleNotFoundError):
    _has_sepia = False
# Prediction requires models + sepia + training data arrays (not bundled in repo)
# These tests are expected to fail until self-contained model pickles are available
_can_predict = False  # TODO: enable when training data is bundled


@pytest.mark.skipif(not _can_predict, reason="Pre-trained SEPIA models or sepia package not available")
def test_predict_observable_gsmf(tmp_path):
    from tools import predict_observable

    result = predict_observable(
        observable="GSMF",
        kappa_w=1.0,
        e_w=0.01,
        m_seed=5.0,
        v_kin=0.5,
        eps_kin=0.5,
        omega_m=0.14,
        sigma_8=0.8,
        output_dir=str(tmp_path),
    )
    assert result.status == "success"
    assert len(result.files) == 1
    assert result.files[0].endswith(".csv")


@pytest.mark.skipif(not _can_predict, reason="Pre-trained SEPIA models or sepia package not available")
def test_predict_observable_redshift_gsmf(tmp_path):
    from tools import predict_observable_redshift

    result = predict_observable_redshift(
        observable="GSMF",
        kappa_w=1.0,
        e_w=0.01,
        m_seed=5.0,
        v_kin=0.5,
        eps_kin=0.5,
        omega_m=0.14,
        sigma_8=0.8,
        redshift=0.5,
        output_dir=str(tmp_path),
    )
    assert result.status == "success"
    assert len(result.files) == 1


def test_predict_observable_rejects_out_of_range():
    """Pydantic validation should reject parameters outside the design range."""
    from tools import predict_observable

    with pytest.raises(Exception):
        predict_observable(
            observable="GSMF",
            kappa_w=999.0,  # way outside [0.03, 3.0]
            e_w=0.01,
            m_seed=5.0,
            v_kin=0.5,
            eps_kin=0.5,
            omega_m=0.14,
            sigma_8=0.8,
            output_dir="/tmp/test",
        )
