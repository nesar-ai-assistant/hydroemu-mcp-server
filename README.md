# hydroemu-mcp-server

An MCP server that exposes HACC cosmological hydrodynamic simulation
emulators — pre-trained SEPIA Gaussian Process models — as tools for any
LLM agent.

## The one idea this repo teaches

> **The science code stays in usual Python. The MCP wrapper only publishes it.**

- `tools/` is an ordinary science package. It never imports MCP. The emulator
  tools live in `tools/hydro_tools.py`; the core SEPIA wrapper is in
  `tools/emulator.py`.
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
models/                           Pre-trained SEPIA pickles (copy from CosmoHydro/models/)
tools/
  emulator.py                    Core SEPIA wrapper: lazy load, predict, redshift interpolation
  hydro_tools.py                 The 5 MCP tool functions + ArtifactResult contract
  __init__.py                    __all__ — ONLY these names become tools
mcp_server/                      Generic drop-in wrapper (FastMCP)
tests/test_tools.py              Tools tested as plain Python, no MCP needed
docs/mcp-clients.md              Multi-client setup guide
```

## Parameters

7 parameters total (5 subgrid + 2 cosmology):

| Parameter | Symbol | Range | Units |
|-----------|--------|-------|-------|
| AGN wind coupling | κ_w | [0.03, 3.0] | — |
| AGN energy efficiency | e_w | [0.001, 0.1] | — |
| BH seed mass | M_seed | [0.5, 50.0] | 10⁶ M☉ |
| Kinetic feedback velocity | v_kin | [0.1, 1.0] | 10⁴ km/s |
| Kinetic feedback efficiency | ε_kin | [0.1, 1.0] | 10¹ |
| Matter density | ω_m | [0.12, 0.155] | — |
| Fluctuation amplitude | σ₈ | [0.7, 0.9] | — |

Design: 110 simulations (400 Mpc/h boxes) from a Latin hypercube design.

## Observables

| Observable | Description | Snapshots | z range |
|------------|-------------|-----------|---------|
| GSMF | Galaxy Stellar Mass Function | 11 | 0–2 |
| HMF | Halo Mass Function | 11 | 0–2 |
| fGas | Cluster Gas Fraction | 7 | 0–1.0 |
| CGD | Cluster Gas Density Profile | 5 | 0–0.5 |
| CGED | Cluster Gas Electron Density Profile | 5 | 0–0.5 |
| CPP | Cluster Gas Pressure Profile | 5 | 0–0.5 |
| CTP | Cluster Gas Temperature Profile | 5 | 0–0.5 |
| CEP | Cluster Gas Entropy Profile | 5 | 0–0.5 |
| CEEP | Cluster Electron Entropy Profile | 5 | 0–0.5 |
| CMP | Cluster Gas Metallicity Profile | 5 | 0–0.5 |
| CYP | Cluster Compton-y (tSZ) Profile | 5 | 0–0.5 |

## Tools

| tool | what it does |
|------|-------------|
| `list_observables()` | list all 11 emulated observables with metadata |
| `describe_parameters()` | the 7-parameter design space with ranges |
| `predict_observable(...)` | predict any observable at z=0, write CSV |
| `predict_observable_redshift(...)` | predict at arbitrary z (interpolated) |
| `plot_observable_comparison(...)` | two-panel figure: observable + ratio |

Two conventions worth copying into any science MCP server:

1. Every tool returns `{status, files, message, metadata}` (`ArtifactResult`).
2. Arrays move between tools **as file paths**, never through the agent's
   context window.

## Install

```bash
conda create -n hydroemu python=3.12 -y
conda activate hydroemu
pip install -e ".[dev]"
pytest                        # tests pass without SEPIA models (fixture data)
```

### Pre-trained models

Copy the trained SEPIA pickles from `CosmoHydro/models/` into the `models/`
directory:

```bash
cp -r /path/to/CosmoHydro/models/GSMF_multiz models/
cp -r /path/to/CosmoHydro/models/HMF_multiz models/
# ... etc for each observable
```

Without models, `list_observables()`, `describe_parameters()`, and
`plot_observable_comparison()` still work; only `predict_observable` and
`predict_observable_redshift` require the pickles.

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
- All SEPIA imports are **lazy** (inside functions, not at module scope)
- Models are loaded on first use and cached for subsequent calls
