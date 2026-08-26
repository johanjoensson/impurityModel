"""TOML input format for the ``spectra`` / ``selfenergy`` / ``susceptibility`` calculations.

Three modules, split by layer so the leaves stay importable from outside this package
(notably from ``impurityModel_interface``, which drives the solver from RSPt's ``green.inp``
and needs ``[environment]`` without pulling in argparse or the solver drivers):

===================  ======  =============================================================
module               layer   may import
===================  ======  =============================================================
``schema``           leaf    stdlib, :mod:`impurityModel.ed.config`
``capabilities``     leaf    stdlib
``reader``           leaf    ``schema``, ``capabilities``, :mod:`impurityModel.ed.h0_format`
``build``            CLI     :mod:`impurityModel.ed.model` and the calculation drivers
===================  ======  =============================================================

Only :mod:`~impurityModel.inputformat.build` sits above the solver; importing anything else
here must never drag in MPI, the drivers, or argparse.
"""
