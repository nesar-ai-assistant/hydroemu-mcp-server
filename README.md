# hydroemu-mcp-server

An MCP server that exposes HACC cosmological hydrodynamic simulation
emulators as tools for any LLM agent.

Powered by the [`cosmohydro_emu`](https://github.com/nesar/cosmohydro_emu)
package — pre-trained SEPIA Gaussian Process models for 14 summary
statistics from the CRK-HACC CosmoHydro simulation suite.

## The one idea this repo teaches

> **The science code stays in usual Python. The MCP wrapper only publishes it.**

- `tools/` is an ordinary science package. It never imports MCP. The emulator
  tools live in `tools/hydro_tools.py`; the emulator backend is provided by
  the `cosmohydro_emu` package (lazily imported inside each tool function).
- `mcp_server/` is a ~70-line generic wrapper. It reads one line of config from
  `pyproject.toml`, imports the science package, and registers every function
  listed in its `__all__` as an MCP tool.

```toml
[tool.mcp-server]
tool_modules = ["tools"]
```

Your type hints, Pydantic `Field` constraints, and docstrings become the tool
schema agents see. To build your own server: drop your modules into `tools/`
(or point that one config line at your own package), list the public functions
in `__all__`, done.

## Layout

```
tools/
  hydro_tools.py                 The 5 MCP tool functions + ArtifactResult contract
  __init__.py                    __all__ — ONLY these names become tools
mcp_server/                      Generic drop-in wrapper (FastMCP)
tests/test_tools.py              Tools tested as plain Python, no MCP needed
docs/mcp-clients.md              Multi-client setup guide
```

## Parameters

Most statistics use 7 parameters (5 subgrid + 2 cosmology). Gravity-only
statistics (Pk_GO) use only the 2 cosmology parameters.

| Parameter | Symbol | Range | Units |
|-----------|--------|-------|-------|
| AGN wind coupling | κ_w | [2.0, 4.0] | — |
| AGN energy efficiency | e_w | [0.2, 1.0] | — |
| BH seed mass | M_seed | [0.6, 2.0] | 10⁶ M☉ |
| Kinetic feedback velocity | v_kin | [0.1, 1.2] | 10⁴ km/s |
| Kinetic feedback efficiency | ε_kin | [0.02, 1.2] | 10¹ |
| Matter density | ω_m | [0.12, 0.155] | — |
| Fluctuation amplitude | σ₈ | [0.7, 0.9] | — |

Design: 110 simulations (400 Mpc/h boxes) from a Latin hypercube design.

## Observables

| Observable | Description | Category | Params | z range |
|------------|-------------|----------|--------|---------|
| GSMF | Galaxy Stellar Mass Function | summary | 7 | 0–2 |
| HMF | Halo Mass Function | summary | 7 | 0–2 |
| fGas | Cluster Gas Fraction | summary | 7 | 0–1.0 |
| Pk-ratio | Matter Power Spectrum Suppression | summary | 7 | 0–2 |
| CSFR | Cosmic Star Formation Rate | summary | 7 | single |
| CGD | Cluster Gas Density Profile | profile | 7 | 0–0.5 |
| CGED | Cluster Gas Electron Density Profile | profile | 7 | 0–0.5 |
| CPP | Cluster Gas Pressure Profile | profile | 7 | 0–0.5 |
| CTP | Cluster Gas Temperature Profile | profile | 7 | 0–0.5 |
| CEP | Cluster Gas Entropy Profile | profile | 7 | 0–0.5 |
| CEEP | Cluster Electron Entropy Profile | profile | 7 | 0–0.5 |
| CMP | Cluster Gas Metallicity Profile | profile | 7 | 0–0.5 |
| CYP | Cluster Compton-y (tSZ) Profile | profile | 7 | 0–0.5 |
| Pk_GO | Gravity-Only Matter Power Spectrum | gravity_only | 2 | 0–2 |

## Tools

| tool | what it does |
|------|-------------|
| `list_observables()` | list all 14 emulated observables with metadata |
| `describe_parameters(stat_name?)` | parameter space with ranges (7 or 2 depending on stat) |
| `predict_observable(stat_name, params..., z)` | predict any observable at any z, write CSV |
| `plot_prediction(stat_name, params..., z)` | single-panel plot with 2σ uncertainty band |
| `plot_observable_comparison(files, stat_name)` | two-panel figure: observable + ratio |

Two conventions worth copying into any science MCP server:

1. Every tool returns `{status, files, message, metadata}` (`ArtifactResult`).
2. Arrays move between tools **as file paths**, never through the agent's
   context window.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                        # tests pass — fixture data for plots, real models for predictions
```

The `cosmohydro_emu` package (and its SEPIA GP backend) is installed
automatically as a dependency. All trained models are shipped inside
`cosmohydro_emu` — no manual model copying needed.

### Related: cosmohydro_emu Python package

The standalone [`cosmohydro_emu`](https://github.com/nesar/cosmohydro_emu)
package provides the same SEPIA emulators as a pip-installable Python library
(with additional statistics: Pk suppression, CSFR, and gravity-only Pk).
This MCP server wraps the same underlying models for agent access.

## Run the server

**Streamable HTTP** — the server is a visible process with a URL:

```bash
python -m mcp_server --transport streamable-http --port 8000
```

Clients connect to `http://127.0.0.1:8000/mcp`. Stop the server with
**Ctrl+C** (Ctrl+Z only suspends it, leaving the port taken — if that
happens, just start the server again: it detects a leftover `mcp_server`
holding the port and clears it automatically).

To use this server from Claude Code, the Claude desktop app, Codex, Cursor,
or any other MCP client — see [`docs/mcp-clients.md`](docs/mcp-clients.md);
a checked-in `.mcp.json` already wires it into Claude Code.

## Architecture

This server follows the same architecture as
[spectra-mcp-server](https://github.com/HEP-KE/spectra-mcp-server):

- `mcp_server/` is a **generic drop-in** MCP wrapper (copy between repos)
- `tools/` contains the domain-specific science functions
- `pyproject.toml` `[tool.mcp-server]` config wires them together
- All `cosmohydro_emu` imports are **lazy** (inside functions, not at module scope)
- Emulators are loaded on first use per tool call

## Emulator Package

The emulator backend is [`cosmohydro_emu`](https://github.com/nesar/cosmohydro_emu),
which ships:

- Pre-trained SEPIA GP models for all 14 statistics
- Training data arrays (parameter designs, x-grids, redshifts)
- Metadata registry (plot info, parameter ranges, output transforms)
- Redshift interpolation between trained snapshot models

See the [cosmohydro_emu documentation](https://github.com/nesar/cosmohydro_emu)
for the full Python API.
