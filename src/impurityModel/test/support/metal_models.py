r"""Synthetic metals for the truncation measurements: Anderson models on semicircular stars.

FCC Ni (15-16 baths) is a poor test of how the Green's-function basis is truncated, because its *ground
state* is already held by the determinant cap (every class stops at the cap with a residual PT2 of 1e-5 and
no progress for several cycles), so a basis comparison there mixes ground-state truncation, Green's-function
truncation and bath discretisation. These models separate them: the ground state converges well below any cap
that binds the Green's-function units, and the exact self-energy is known at ``U = 0``.

One model is ``n_orb`` degenerate impurity orbitals, each hybridised with its own ``n_b``-level
Gauss-Chebyshev star (:func:`~impurityModel.test.support.aim_fixtures.semicircle_star`) of half bandwidth
``D`` and hybridisation strength ``v``, with a Hubbard-Kanamori interaction on the impurity. It is written as a
spinless ``.h0`` file plus a TOML input (``spin = "degenerate"``), so it runs through ``impurityModel run`` and
through :mod:`freeze_replay` alike:

* ``n_orb = 1``: the single-band SIAM of ``examples/semicircular_siam`` (a Kondo resonance at the Fermi level);
* ``n_orb = 3``: a Hund's metal.

``[double_counting.nominal]`` puts the impurity level at the particle-hole symmetric point, so at ``U = J = 0``
the self-energy is **identically zero** and any nonzero value the solver returns is truncation. The excitation
budget is switched off: with ``V`` of the order of the bandwidth the ground state carries many bath excitations
and the default budget of 4 makes Sigma acausal even at ``U = 0`` (see the example's comment).
"""

from pathlib import Path

import numpy as np

from impurityModel.ed.h0_format import write_h0_file
from impurityModel.test.support.aim_fixtures import semicircle_star

#: The TOML input :func:`write_metal` fills in; ``{{ }}`` is a literal brace pair for ``str.format``.
TOML_TEMPLATE = """\
[format]
version = [1, 0]

[units]
energy = "Ry"

[hamiltonian.file]
path = "{h0_name}"
spin = "degenerate"

[[shell]]
n_orbitals = {n_orb}
role = "valence"
nominal_occupation = {nominal}

[interaction.kanamori]
U = {U!r}
J = {J!r}

[double_counting.nominal]

[temperature]
tau = {tau!r}

[selfenergy]
cluster = "{name}"

[selfenergy.real_axis]
enabled = true
mesh = {{ min = {w_min!r}, max = {w_max!r}, n = {n_w} }}
broadening = {broadening!r}

[selfenergy.matsubara]
enabled = {matsubara}
n_points = {n_matsubara}
"""


def metal_h0(n_orb, n_b, D, v):
    """The spinless one-particle matrix: ``n_orb`` impurity orbitals first, then ``n_b`` bath levels per orbital.

    Orbital ``a``'s bath levels sit at ``n_orb + a * n_b + k`` with the Gauss-Chebyshev levels and couplings of
    :func:`~impurityModel.test.support.aim_fixtures.semicircle_star`, so ``sum_k V_k^2 = v^2`` for every orbital.
    The impurity level is zero; the double counting shifts it.
    """
    levels, couplings = semicircle_star(n_b, D, v)
    h = np.zeros((n_orb * (1 + n_b),) * 2)
    for a in range(n_orb):
        for k in range(n_b):
            b = n_orb + a * n_b + k
            h[b, b] = levels[k]
            h[a, b] = h[b, a] = couplings[k]
    return h


def write_metal(
    directory,
    name,
    *,
    n_orb=1,
    n_b=14,
    D=0.5,
    v=1.0,
    U=0.5,
    J=0.0,
    tau=2e-4,
    nominal_occupation=None,
    n_matsubara=16,
    real_mesh=(-1.5, 1.5, 301),
    broadening=0.02,
):
    """Write ``<name>.h0`` and ``<name>.toml`` into ``directory``; returns the TOML path.

    ``nominal_occupation`` defaults to ``n_orb`` (half filling). Both axes are on by default because
    :func:`freeze_replay.grade` compares the Matsubara and the real-axis self-energy. ``U = J = 0`` is the
    exactly solvable leg (Sigma identically zero).
    """
    if n_orb < 1 or n_b < 2 or n_b % 2:
        # An odd star has a level at zero: the input pipeline splits the bath into valence and conduction by the
        # sign of the level, so a zero level lands on one side and the model loses its particle-hole symmetry.
        raise ValueError(f"need n_orb >= 1 and an even n_b >= 2, got n_orb={n_orb}, n_b={n_b}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h0_name = f"{name}.h0"
    write_h0_file(
        directory / h0_name,
        metal_h0(n_orb, n_b, D, v),
        impurity_orbitals={0: list(range(n_orb))},
        unit="Ry",
    )
    w_min, w_max, n_w = real_mesh
    toml = TOML_TEMPLATE.format(
        h0_name=h0_name,
        name=name,
        n_orb=n_orb,
        nominal=n_orb if nominal_occupation is None else nominal_occupation,
        U=float(U),
        J=float(J),
        tau=float(tau),
        w_min=float(w_min),
        w_max=float(w_max),
        n_w=int(n_w),
        broadening=float(broadening),
        matsubara="true" if n_matsubara > 0 else "false",
        n_matsubara=int(n_matsubara),
    )
    path = directory / f"{name}.toml"
    path.write_text(toml, encoding="utf-8")
    return path
