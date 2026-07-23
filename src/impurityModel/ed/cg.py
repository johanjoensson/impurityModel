"""Block BiCGSTAB linear solver — thin Python wrapper.

All BiCGSTAB business logic (the rank-deflating entry point and the ``_block_bicgstab_core``
inner iteration, over both the dense array and the ``ManyBodyState`` representations)
lives in Cython in :mod:`impurityModel.ed.BiCGSTAB`. This module only re-exports the public
entry point (and the internal core, for backwards compatibility) so the existing
``from impurityModel.ed.cg import ...`` import paths keep working.
"""

from impurityModel.ed.BiCGSTAB import _block_bicgstab_core, block_bicgstab

__all__ = ["_block_bicgstab_core", "block_bicgstab"]
