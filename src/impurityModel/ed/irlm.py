"""Implicitly Restarted Block Lanczos (IRLM) — thin Python wrapper.

All IRLM business logic (the EA16 path-agnostic core ``_irlm_core``, the array /
ManyBodyState entry points, locking, explicit purging, and result assembly) lives in
Cython in :mod:`impurityModel.ed.BlockLanczos`. This module only re-exports the public
dispatching entry point (and the internal helpers, for backwards compatibility) so the
existing ``from impurityModel.ed.irlm import ...`` import paths keep working.
"""

from impurityModel.ed.BlockLanczos import (
    implicitly_restarted_block_lanczos as implicitly_restarted_block_lanczos_cy,
    _implicitly_restarted_block_lanczos_array,
    _implicitly_restarted_block_lanczos_manybody,
    _irlm_core,
    _assemble_results,
)

__all__ = [
    "implicitly_restarted_block_lanczos_cy",
    "_implicitly_restarted_block_lanczos_array",
    "_implicitly_restarted_block_lanczos_manybody",
    "_irlm_core",
    "_assemble_results",
]
