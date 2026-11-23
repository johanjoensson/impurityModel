r"""The second-order growth score of the pruned Green's-function growth (``gf_pt2_tol``).

A frontier determinant ``f`` (discovered, not yet in ``P``) gets

.. math::

    s_f = \max_k \sum_{c} \frac{|\rho_{fc}(z_k)|^2}{|z_k - H_{ff}|\,\|G_P(z_k)\|},
    \qquad \rho = H X(z),\quad X = (z - H_{PP})^{-1} Q_1,

over a uniform mesh ``z_k`` across the unit's own real-frequency evaluation window at the run's own broadening:
the coherent second-order weight of ``f`` in the resolvent, normalised by the size of ``G`` at that frequency, so
the summed score of the rows left out is a budget on the relative error of ``G``. See
``doc/plans/pt2_growth_rule.md`` for the measurements.

``rho`` costs no extra application of ``H``: the :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`
recursion is linear in the Lanczos block, so stepping it with the matvec output ``w_j = H v_j`` (every K row, which
the discovered matrix returns anyway) accumulates ``H X`` directly. The scores read the accumulator as it stood
before the block being decided is folded (its coefficients are known only after the step): a row first reached at
block ``j`` scores zero and is scored a scored block later; a deferred row is never banned.

Rank-local except where stated: the verdicts that steer collectives (``due``, the mesh drop, the cut) are functions
of replicated state.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.gf_primitives import calc_G
from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin
from impurityModel.ed.manybody_basis import collective_mass_cutoff

#: Points of the uniform scoring mesh over the unit's real-frequency window.
PT2_MESH_POINTS = 31
#: Score the frontier on every this-many-th block (the accumulator is stepped on every block regardless). Scoring every
#: block keeps the relative error of ``G`` inside ``gf_pt2_tol`` on everything measured; every 4th block is about
#: twice as cheap in growth and reaches 1.6 times it on paramagnetic NiO.
PT2_SCORE_EVERY = 4
#: A mesh point whose share of the frontier's score mass is below this on each of the first
#: :data:`PT2_DROP_BLOCKS` scored blocks that carry mass is dropped from the accumulator (``0``: never drop).
PT2_DROP_SHARE = 1e-3
PT2_DROP_BLOCKS = 10
#: Growth ends once no row has been admitted for as many blocks as it took to reach the last admission, and for at
#: least this many (a test that wants the unstopped run sets it very large).
PT2_QUIET_BLOCKS = 16


def pt2_mesh(eval_meshes, n_points=PT2_MESH_POINTS):
    """``(zs, why)``: the scoring mesh in the frame of the recurrence's coefficients, or ``(None, reason)`` when the
    unit has no real-frequency window to score over.

    ``eval_meshes`` is :func:`~impurityModel.ed.gf_convergence._gf_eval_meshes`'s result: one array per requested
    axis, already in the coefficients' frame (``+e`` for every stacked thermal state). The real axis is the one whose
    points all share one nonzero imaginary part and sit at three or more distinct real parts (a Matsubara axis has
    ``Im z`` varying with the point). The mesh is uniform from the smallest to the largest real part over **all** the
    states' windows, at that imaginary part: nothing is invented, and a Matsubara-only unit has no mesh.
    """
    if not eval_meshes:
        return None, "the unit evaluates no frequency mesh"
    for mesh in eval_meshes:
        z = np.asarray(mesh, dtype=complex).reshape(-1)
        if z.size < 3:
            continue
        im = z.imag
        if not (np.ptp(im) <= 1e-12 * max(1.0, float(np.abs(im).max())) and im[0] != 0.0):
            continue
        if len(np.unique(np.round(z.real, 12))) < 3:
            continue
        return np.linspace(z.real.min(), z.real.max(), int(n_points)) + 1j * im[0], None
    return None, "the unit has no real-frequency evaluation mesh"


class GrowthScorer:
    """The accumulator and the scores of one unit's growth pass. **Collective** where noted (``scores``).

    Parameters
    ----------
    zs : array
        The scoring mesh (:func:`pt2_mesh`), in the recurrence's frame.
    tau : float
        The budget of the score mass left out by one admission (the option ``gf_pt2_tol``).
    comm : MPI.Comm or None
    every, drop, drop_blocks
        :data:`PT2_SCORE_EVERY`, :data:`PT2_DROP_SHARE`, :data:`PT2_DROP_BLOCKS`.
    """

    def __init__(self, zs, tau, comm, every=None, drop=None, drop_blocks=None):
        self.zs0 = np.asarray(zs, dtype=complex).reshape(-1)
        self.tau = float(tau)
        self.comm = comm if comm is not None and comm.size > 1 else None
        # Read at construction, not at definition: a test (or a study) can set the module constants.
        self.every = max(1, int(PT2_SCORE_EVERY if every is None else every))
        self.drop = float(PT2_DROP_SHARE if drop is None else drop)
        self.drop_blocks = int(PT2_DROP_BLOCKS if drop_blocks is None else drop_blocks)
        self.acc = None
        self._stash = None
        self._alphas = []
        self._betas = []
        self._beta_prev = None
        #: Recurrence blocks folded into the accumulator (replicated).
        self.blocks = 0
        #: Decisions that scored the frontier, and the block of the last one that admitted a row (replicated).
        self.scored = 0
        self.last_admission = None
        #: Mesh points still in the accumulator, as indices into the full mesh.
        self.kept = np.arange(self.zs0.size)
        self.dropped = False
        self._live = np.zeros(self.zs0.size, dtype=bool)
        self._seen = 0
        #: Rows the rule admitted after the first shell (replicated, running).
        self.admitted = 0
        #: Retained size at the start of the growth: a unit that has not grown past it has nothing to restart on, so
        #: its recurrence must run to its own convergence, never be cut by the quiet stop.
        self.n_start = None
        #: Why the rule was dropped mid-growth (the accumulator did not fit the measured K), or ``None``.
        self.declined = None

    def has_stash(self):
        """Whether a matvec output is waiting to be folded (replicated: set on every unfrozen apply)."""
        return self._stash is not None

    @property
    def zs(self):
        return self.zs0[self.kept]

    def stash(self, w_k):
        """The matvec output ``H v_j`` over every K row of the block about to be decided (kept by reference)."""
        self._stash = w_k

    def fold(self, alpha, beta):
        """Step the accumulator with the stashed block, ``alpha`` and the previous ``beta`` of the step the recurrence
        just accepted; ``beta`` (the ``(k, p)`` factor of its TSQR) is kept for the next fold and for ``||G_P||``."""
        w = self._stash
        self._stash = None
        if w is None:
            return
        if self.acc is None:
            self.acc = ShiftedGalerkin(self.zs, w.shape[0])
        self.acc.step(w, alpha, self._beta_prev)
        self._alphas.append(np.array(alpha, dtype=complex))
        self._betas.append(np.array(beta, dtype=complex))
        self._beta_prev = self._betas[-1]
        self.blocks += 1

    def due(self):
        """Whether the decision now scores the frontier (replicated): an accumulator exists and the block is on the
        cadence. A decision that is not due admits nothing."""
        return self.acc is not None and self.blocks % self.every == 0

    def gnorm(self):
        """``||G_P(z_k)||`` (Frobenius, ``(n_kept,)``): the seed block of ``(z - T)^-1`` from the recurrence so far."""
        n0 = self._alphas[0].shape[0]
        g = calc_G(self._alphas, self._betas, np.eye(n0, dtype=complex), self.acc.zs, 0.0, 0.0)
        return np.linalg.norm(g, axis=(1, 2))

    def scores(self, slots, diag):
        """``(s, mass)``: the score of each frontier row (``slots`` are K slots, ``diag`` their ``H_ff``) and the
        global score mass of the rows per kept mesh point (what the mesh drop reads). **Collective** (``mass``)."""
        acc = self.acc
        gn = self.gnorm()
        n_z = acc.n_z
        sols = acc.solutions()
        ok = slots < acc.n_rows  # a row reached after the last fold carries no mass yet
        num = np.zeros((n_z, len(slots)))
        if ok.any():
            for kz in range(n_z):
                num[kz, ok] = (np.abs(sols[kz][slots[ok]]) ** 2).sum(axis=1)
        den = np.maximum(np.abs(acc.zs[:, None] - np.asarray(diag)[None, :]), 1e-12)
        s_all = num / den / gn[:, None]
        s = s_all.max(axis=0) if len(slots) else np.zeros(0)
        mass = s_all.sum(axis=1)
        if self.comm is not None:
            self.comm.Allreduce(MPI.IN_PLACE, mass, op=MPI.SUM)
        return s, mass

    def cutoff(self, s):
        """The score above which a row is admitted: one global tail cut of mass ``tau``. **Collective.**"""
        return collective_mass_cutoff(s, self.tau, self.comm)

    def note_mass(self, mass):
        """Mesh-point drop from the global per-point mass of a scored decision (replicated, so every rank drops the
        same points): after :data:`PT2_DROP_BLOCKS` scored blocks that carried mass, the points that held a share of
        it below :data:`PT2_DROP_SHARE` on every one of them are removed from the accumulator."""
        total = float(mass.sum())
        if self.drop <= 0.0 or self.dropped or not total > 0.0:
            return
        self._seen += 1
        self._live |= mass / total >= self.drop
        if self._seen >= self.drop_blocks:
            self.dropped = True
            if not self._live.all():
                self.acc.keep(self._live)
                self.kept = self.kept[self._live]

    def quiet(self, retained=None):
        """Whether the growth is over (replicated): the blocks since the last admission (or since the start, when the
        rule admitted nothing) are at least :data:`PT2_QUIET_BLOCKS` and at least the blocks that preceded it, i.e. the
        run is twice as old as its last admission."""
        if self.n_start is not None and retained is not None and retained <= self.n_start:
            return False
        last = 0 if self.last_admission is None else self.last_admission
        return self.blocks - last >= max(PT2_QUIET_BLOCKS, last)

    def release(self):
        """Drop the accumulator, the stash and the coefficient lists (GBs on a large unit); the counters the report reads
        stay."""
        self.acc = None
        self._stash = None
        self._alphas = []
        self._betas = []
        self._beta_prev = None

    def note_decision(self, n_admitted):
        """Book a scored decision (replicated counts)."""
        self.scored += 1
        if n_admitted > 0:
            self.last_admission = self.blocks
            self.admitted += int(n_admitted)
