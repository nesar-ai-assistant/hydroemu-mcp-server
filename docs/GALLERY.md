# Emulator Plots Gallery

All plots are generated from the CosmoHydro training notebooks (02 and 03).
They validate the SEPIA GP emulators against held-out HACC simulations.

## Galaxy / Halo Statistics

### GSMF — Galaxy Stellar Mass Function

| Plot | Description |
|------|-------------|
| ![GSMF Validation](plots/emu_GSMF_multiz_valid_1.png) | **Validation at z=0**: 3 held-out test simulations (solid) vs emulator predictions (dashed + uncertainty band). 16 stellar mass bins from 5×10⁹ to 3×10¹¹ M☉. |
| ![GSMF z-interpolation](plots/emu_GSMF_z_interp.png) | **Redshift interpolation**: Emulator prediction at intermediate redshifts using linear interpolation between trained snapshots. |
| ![GSMF Sensitivity](plots/emu_GSMF_sensi_multiz.png) | **Parameter sensitivity**: GSMF response to σ₈ vs κ_w at multiple redshifts (3-panel). Shows how subgrid and cosmology parameters affect the mass function differently at different epochs. |

### HMF — Halo Mass Function

| Plot | Description |
|------|-------------|
| ![HMF Validation](plots/emu_HMF_multiz_valid_1.png) | **Validation at z=0**: Emulator vs truth for 3 test simulations. |

### fGas — Cluster Gas Fraction

| Plot | Description |
|------|-------------|
| ![fGas Validation](plots/emu_fGas_multiz_valid_1.png) | **Validation at z=0**: Gas fraction as a function of halo mass for test simulations. |
| ![fGas Multi-z](plots/emu_fGas_multiz.png) | **Multi-redshift ensemble**: All 110 simulations at multiple snapshot redshifts, showing the spread from subgrid parameter variation. |

## Matter Power Spectrum

| Plot | Description |
|------|-------------|
| ![Pk Validation](plots/emu_Pk_valid_1.png) | **P(k) suppression validation**: Ratio of hydrodynamic to gravity-only power spectrum at z=0. Shows baryonic feedback effects on large-scale structure. |
| ![Pk Sensitivity](plots/emu_Pk_sensi.png) | **Parameter sensitivity**: Per-parameter contribution to P(k) suppression. Shows which subgrid parameters dominate at different k-scales. |
| ![Pk Design](plots/emu_Pk_design.png) | **Design-colored P(k)**: Suppression curves colored by two design parameters, revealing the parameter-dependence of baryon effects. |
| ![Pk Boosted vs Suppressed](plots/emu_Pk_boosted_vs_suppressed_4x2.png) | **Extreme cases**: Simulations with the most boosted vs most suppressed P(k), 4×2 panel comparing different physical regimes. |

## Experimental Design

| Plot | Description |
|------|-------------|
| ![Design Space](plots/emu_snapshot_z.png) | **Parameter design scatter matrix**: 110-point Latin hypercube design in the 7D parameter space (5 subgrid + 2 cosmology). |

## Cluster Thermodynamic Profiles

### Multi-panel Overviews

| Plot | Description |
|------|-------------|
| ![All Profiles Multi-z](plots/prof_profiles_all_multiz.png) | **All 8 cluster profiles at multiple redshifts**: CGD, CGED, CPP, CTP, CEP, CEEP, CMP, CYP — simulation ensemble shown as colored bands across redshift snapshots. |
| ![Profiles Validation](plots/prof_profiles_multiz_validation.png) | **Profile validation**: Emulator vs truth for cluster gas density and electron density profiles at z=0. |
| ![Profiles z-interpolation](plots/prof_profiles_multiz_redshift_interp.png) | **Redshift interpolation**: Profile predictions at intermediate redshifts. |

### Comprehensive Diagnostics

| Plot | Description |
|------|-------------|
| ![All Profiles Validation z=0](plots/prof_profiles_all_validation_z0.png) | **All profiles, validation at z=0**: Every profile type validated against 3 held-out test simulations. |
| ![All Profiles Sensitivity z=0](plots/prof_profiles_all_sensitivity_z0.png) | **Per-parameter sensitivity at z=0**: How each of the 7 parameters affects each cluster profile. Critical for understanding degeneracies. |
| ![Profiles z-sensitivity](plots/prof_profiles_all_z_sensitivity.png) | **Redshift sensitivity**: Profile evolution from z=0.5 to z=0 at fiducial parameters. |
| ![Profiles vs Observations](plots/prof_profiles_emu_vs_obs_z0.png) | **Emulator vs Observations**: Cluster profiles compared to observational data from X-ray and SZ measurements. |
| ![Profiles vs Observations (obs only)](plots/prof_profiles_emu_vs_obs_z0_obsonly_1.png) | **Observation overlay**: Same as above but focused on the observational data points and error bars. |

## Cosmic Star Formation Rate (CSFR)

| Plot | Description |
|------|-------------|
| ![CSFR Design](plots/csfr_cell5_1.png) | **Design-space scatter matrix**: Full 7-parameter Latin hypercube with CSFR-specific view. |
| ![CSFR Ensemble](plots/csfr_cell7_2.png) | **CSFR ensemble**: All 110 simulations showing star formation rate density vs scale factor. |
| ![CSFR Raw](plots/csfr_cell8_1.png) | **Single simulation CSFR**: Raw cosmic star formation rate for one simulation. |
| ![CSFR Processed](plots/csfr_cell10_6.png) | **Processed CSFR data**: After NaN interpolation and mass cuts. |
| ![CSFR PCA basis](plots/csfr_cell12.png) | **PCA basis function**: Leading principal component from SEPIA decomposition. |
| ![CSFR Validation](plots/csfr_cell14_2.png) | **Validation**: Emulator predictions vs held-out test simulations for CSFR. |
| ![CSFR Sensitivity](plots/csfr_cell16_1.png) | **Parameter sensitivity**: Per-parameter contribution to CSFR variation. |
