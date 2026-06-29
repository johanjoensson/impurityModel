"""Thick-Restart Block Lanczos (TRLM) — thin Python wrapper.

All TRLM business logic (the path-agnostic ``_trlm_core``, the array / ManyBodyState entry
points, and the width-aware thick-restart loop) lives in Cython in
:mod:`impurityModel.ed.BlockLanczos`. This module only re-exports the public dispatching
entry point so the existing ``from impurityModel.ed.trlm import ...`` import paths keep
working.
"""

from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos

__all__ = ["thick_restart_block_lanczos"]
