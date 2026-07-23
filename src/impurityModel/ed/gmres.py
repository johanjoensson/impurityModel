"""Restarted block-GMRES linear solver — thin Python wrapper.

All GMRES business logic (the rank-deflating entry point, the block Arnoldi restart
cycles, over both the dense array and the ``ManyBodyState`` representations) lives
in Cython in :mod:`impurityModel.ed.GMRES`. This module only re-exports the public entry
point so the import surface matches the ``cg.py`` / ``irlm.py`` / ``trlm.py``
thin-wrapper arrangement.
"""

from impurityModel.ed.GMRES import block_gmres

__all__ = ["block_gmres"]
