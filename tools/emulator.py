# This module has been removed. The cosmohydro_emu package now provides
# all emulator functionality. This file is kept empty to avoid import errors
# during the transition.
raise ImportError(
    "tools.emulator has been replaced by the cosmohydro_emu package. "
    "Use 'from cosmohydro_emu import load_emulator' instead."
)
