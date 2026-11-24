r"""One-body operators of the correlated impurity block, in the model's input basis.

RSPt holds ``S_z``, ``L_z``, ``S_+``, ``L_+`` (and optionally the dipole ``Dxyz``) for every correlated
cluster in the same basis as ``ham``/``hyb``/``sig``. Everything the solver needs to know about spin and
orbital structure -- spin pairing, S_z sectors, Casimir reports, dipole transition operators -- is a
function of those matrices. :class:`LocalOperators` carries them; :meth:`LocalOperators.from_spherical`
rebuilds the same set from ``rot_to_spherical`` and the shell layout when nothing else is available.

Layering: bottom layer, numpy only.

All matrices are ``(n, n)`` with ``O_input = R O_spherical R^dagger`` semantics, the dipole is
``(3, n, n)`` Cartesian (``x, y, z``). RSPt's circular ``Dpm`` is derived, see
:meth:`LocalOperators.dipole_circular`.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

__all__ = ["LocalOperators", "SiteFrame", "single_particle_lsj_matrices", "shell_operator_matrices", "site_frames"]

_SOURCES = ("rspt", "file", "spherical", "index")
_FRAMES = ("global", "local")


def single_particle_lsj_matrices(l):
    r"""Single-particle ``L`` and ``S`` operator matrices of one spin-doubled shell in the spherical basis.

    Layout: ``2*(2l+1)`` orbitals, the first ``2l+1`` spin-down (:math:`m_s=-1/2`, ``ml=-l..l``), the next
    ``2l+1`` spin-up.

    Returns
    -------
    (lz, lplus, lminus, sz, splus, sminus) : tuple of np.ndarray
        Each of shape ``(2*(2l+1), 2*(2l+1))``.
    """
    n = 2 * l + 1
    mls = np.arange(-l, l + 1)
    llp1 = l * (l + 1)
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    lz = np.diag(np.concatenate((mls, mls)).astype(float)).astype(complex)
    sz = np.diag(np.concatenate((-0.5 * np.ones(n), 0.5 * np.ones(n)))).astype(complex)
    lplus_block = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in mls[:-1]], k=-1)
    lplus = np.block([[lplus_block, zeros], [zeros, lplus_block]]).astype(complex)
    lminus = lplus.conj().T
    splus = np.block([[zeros, zeros], [eye, zeros]]).astype(complex)
    sminus = splus.conj().T
    return lz, lplus, lminus, sz, splus, sminus


def _shell_slices(shell_ls):
    out, start = [], 0
    for l in shell_ls:
        out.append(slice(start, start + 2 * (2 * l + 1)))
        start += 2 * (2 * l + 1)
    return out


def shell_operator_matrices(rot_to_spherical, shell_ls):
    r"""Per shell ``a``, the six matrices ``R_a M R_a^dagger`` (``lz, l+, l-, sz, s+, s-``) on the whole block.

    ``R_a`` is the shell's column block of ``rot_to_spherical``; it may be rectangular (a partial shell), in
    which case the matrices are the operators projected onto the kept space, as RSPt computes them.
    """
    rot = np.asarray(rot_to_spherical, dtype=complex)
    out = []
    for l, cols in zip(shell_ls, _shell_slices(shell_ls)):
        R_a = rot[:, cols]
        out.append([R_a @ m @ R_a.conj().T for m in single_particle_lsj_matrices(l)])
    return out


@dataclass(frozen=True)
class LocalOperators:
    r"""One-body operators of the impurity block, all in the model's input basis.

    Attributes
    ----------
    sz, s_plus, lz, l_plus : ndarray or None
        ``(n, n)``. Any may be absent (a basis with spin ordering only has ``sz``/``s_plus``).
    s_axis : ndarray or None
        ``(n, n)`` spin component along the quantisation axis when it differs from ``z``
        (RSPt's ``msaxis``); ``None`` means ``sz`` is the one.
    dipole : ndarray or None
        ``(3, n, n)`` Cartesian dipole matrix over the whole block (core-valence block included).
    symmetry_ops : tuple of (ndarray, bool)
        Supplied symmetry operations ``(U, antiunitary)``; verified against the data, never imposed.
    source : str
        ``"rspt"``, ``"file"``, ``"spherical"`` or ``"index"``.
    frame : str
        ``"global"`` (RSPt's Cartesian frame) or ``"local"`` (the shell's own frame).
    """

    sz: Optional[np.ndarray] = None
    s_plus: Optional[np.ndarray] = None
    lz: Optional[np.ndarray] = None
    l_plus: Optional[np.ndarray] = None
    s_axis: Optional[np.ndarray] = None
    dipole: Optional[np.ndarray] = None
    symmetry_ops: Tuple = ()
    source: str = "spherical"
    frame: str = "local"

    def __post_init__(self):
        if self.source not in _SOURCES:
            raise ValueError(f"source must be one of {_SOURCES}, got {self.source!r}")
        if self.frame not in _FRAMES:
            raise ValueError(f"frame must be one of {_FRAMES}, got {self.frame!r}")
        n = None
        for name in ("sz", "s_plus", "lz", "l_plus", "s_axis"):
            m = getattr(self, name)
            if m is None:
                continue
            m = np.asarray(m, dtype=complex)
            if m.ndim != 2 or m.shape[0] != m.shape[1]:
                raise ValueError(f"{name} must be square, got shape {m.shape}")
            if n is not None and m.shape[0] != n:
                raise ValueError(f"{name} has size {m.shape[0]}, expected {n}")
            n = m.shape[0]
            object.__setattr__(self, name, m)
        if self.dipole is not None:
            d = np.asarray(self.dipole, dtype=complex)
            if d.ndim != 3 or d.shape[0] != 3 or d.shape[1] != d.shape[2] or (n is not None and d.shape[1] != n):
                raise ValueError(f"dipole must have shape (3, {n}, {n}), got {d.shape}")
            object.__setattr__(self, "dipole", d)
        ops = []
        for u, anti in self.symmetry_ops:
            u = np.asarray(u, dtype=complex)
            if u.ndim != 2 or u.shape[0] != u.shape[1] or (n is not None and u.shape[0] != n):
                raise ValueError(f"symmetry operation has shape {u.shape}, expected ({n}, {n})")
            ops.append((u, bool(anti)))
        object.__setattr__(self, "symmetry_ops", tuple(ops))

    @property
    def n(self) -> Optional[int]:
        """The block size, ``None`` for an empty set."""
        for name in ("sz", "s_plus", "lz", "l_plus", "s_axis"):
            m = getattr(self, name)
            if m is not None:
                return m.shape[0]
        return None if self.dipole is None else self.dipole.shape[1]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_spherical(cls, rot_to_spherical, shell_ls, *, site_rotation=None):
        r"""The operators rebuilt from the spherical rotation: ``O = sum_a R_a O(l_a) R_a^dagger``.

        ``rot_to_spherical`` is the ``(n, sum_a 2(2 l_a + 1))`` rotation (``O_input = R O_sph R^dagger``).
        A partial shell gives the projected operators. ``site_rotation`` is an optional ``(3, 3)`` proper
        rotation taking the shell's local frame to the global one; it rotates the spin and orbital vectors
        (``S_z`` and ``S_+`` are then the global components) and marks the set ``frame="global"``.
        """
        rot = np.asarray(rot_to_spherical, dtype=complex)
        shells = shell_operator_matrices(rot, shell_ls)
        n = rot.shape[0]
        total = [sum((s[k] for s in shells), np.zeros((n, n), dtype=complex)) for k in range(6)]
        lz, lp, lm, sz, sp, sm = total
        ops = cls(sz=sz, s_plus=sp, lz=lz, l_plus=lp, source="spherical", frame="local")
        if site_rotation is None:
            return ops
        r = np.asarray(site_rotation, dtype=float)
        if r.shape != (3, 3) or not np.allclose(r @ r.T, np.eye(3), atol=1e-8) or np.linalg.det(r) < 0:
            raise ValueError("site_rotation must be a proper 3x3 rotation")
        return ops._in_global_frame(r)

    @classmethod
    def from_spin_ordering(cls, n_imp, ordering="down_first"):
        """Spin operators only, for a basis whose sole known structure is the dn/up ordering."""
        if n_imp % 2:
            raise ValueError("a spin-doubled block has an even number of orbitals")
        h = n_imp // 2
        sz = np.zeros((n_imp, n_imp), dtype=complex)
        sp = np.zeros((n_imp, n_imp), dtype=complex)
        if ordering == "down_first":
            dn, up = np.arange(h), np.arange(h, n_imp)
        elif ordering == "up_first":
            dn, up = np.arange(h, n_imp), np.arange(h)
        elif ordering == "interleaved":  # dn, up, dn, up, ...
            dn, up = np.arange(0, n_imp, 2), np.arange(1, n_imp, 2)
        else:
            raise ValueError(f"unknown spin ordering {ordering!r}")
        sz[dn, dn] = -0.5
        sz[up, up] = 0.5
        sp[up, dn] = 1.0
        return cls(sz=sz, s_plus=sp, source="index")

    # ------------------------------------------------------------------ transformations
    def rotated(self, Q):
        r"""Every member in the basis ``Q``: ``O -> Q^dagger O Q``; antiunitary ``U -> Q^dagger U Q^*``."""
        Q = np.asarray(Q, dtype=complex)
        qd = Q.conj().T

        def rot(m):
            return None if m is None else qd @ m @ Q

        dip = None if self.dipole is None else np.stack([qd @ d @ Q for d in self.dipole])
        syms = tuple((qd @ u @ (Q.conj() if anti else Q), anti) for u, anti in self.symmetry_ops)
        return replace(
            self,
            sz=rot(self.sz),
            s_plus=rot(self.s_plus),
            lz=rot(self.lz),
            l_plus=rot(self.l_plus),
            s_axis=rot(self.s_axis),
            dipole=dip,
            symmetry_ops=syms,
        )

    def _in_global_frame(self, r):
        """Spin/orbital components along the global axes for a local frame rotated by ``r``."""
        sv, lv = self.spin_vector(), self.orbital_vector()
        out = {}
        if sv is not None:
            sg = np.tensordot(r, sv, axes=(1, 0))  # global component i = sum_j r[i, j] S_local_j
            out["sz"], out["s_plus"] = sg[2], sg[0] + 1j * sg[1]
            out["s_axis"] = None
        if lv is not None:
            lg = np.tensordot(r, lv, axes=(1, 0))
            out["lz"], out["l_plus"] = lg[2], lg[0] + 1j * lg[1]
        return replace(self, frame="global", **out)

    # ------------------------------------------------------------------ views
    @staticmethod
    def _cartesian(z, plus):
        if z is None or plus is None:
            return None
        x = 0.5 * (plus + plus.conj().T)
        y = (plus - plus.conj().T) / 2j
        return np.stack([x, y, z])

    def spin_vector(self):
        """``(3, n, n)`` ``(S_x, S_y, S_z)``, or ``None`` when ``sz``/``s_plus`` are missing."""
        return self._cartesian(self.sz, self.s_plus)

    def orbital_vector(self):
        """``(3, n, n)`` ``(L_x, L_y, L_z)``, or ``None`` when ``lz``/``l_plus`` are missing."""
        return self._cartesian(self.lz, self.l_plus)

    def dipole_circular(self):
        r"""RSPt's ``Dpm`` triple ``(Dp, Dm, Dz)`` from the Cartesian dipole.

        ``Dp = (Dx + i Dy)/sqrt2``, ``Dm = (Dx - i Dy)/sqrt2`` (``green_locust.F90``); the inverse is
        ``Dx = (Dp + Dm)/sqrt2``, ``Dy = (Dp - Dm)/(i sqrt2)``, see :func:`dipole_from_circular`.
        """
        if self.dipole is None:
            return None
        dx, dy, dz = self.dipole
        s = np.sqrt(0.5)
        return np.stack([s * (dx + 1j * dy), s * (dx - 1j * dy), dz])


@dataclass(frozen=True)
class SiteFrame:
    r"""The proper rotation taking a shell's local frame (that of ``rot_to_spherical``) to the global one.

    ``rotation[i, j]`` is the global component ``i`` of local axis ``j``: a vector operator transforms as
    ``V_global_i = sum_j rotation[i, j] V_local_j``, the convention of :meth:`LocalOperators.from_spherical`'s
    ``site_rotation``. ``residual`` is the largest element of ``L_global - rotation . L_local`` on the shell
    block after the fit: roundoff when the supplied operators are a rotation of the spherical ones, of order
    one when the basis is not what ``rot_to_spherical`` says.
    """

    rotation: np.ndarray
    residual: float

    @property
    def angle(self) -> float:
        """Rotation angle in radians."""
        return float(np.arccos(np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)))

    @property
    def axis(self) -> np.ndarray:
        """Unit rotation axis (``z`` for the identity)."""
        r = self.rotation
        v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
        if self.angle < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        w, vecs = np.linalg.eigh(0.5 * (r + r.T))  # angle pi: the axis is the +1 eigenvector
        return vecs[:, np.argmax(w)]

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.rotation, np.eye(3), atol=1e-8))

    def describe(self) -> str:
        if self.is_identity:
            return f"local frame = global frame (residual {self.residual:.1e})"
        ax = ", ".join(f"{x:+.4f}" for x in self.axis)
        return f"rotated by {np.degrees(self.angle):.3f} deg about ({ax}) relative to the global frame (residual {self.residual:.1e})"


def _nearest_rotation(m):
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def site_frames(supplied, rot_to_spherical, shell_ls):
    r"""Per shell, the :class:`SiteFrame` that maps the spherical rebuild onto ``supplied``'s orbital vector.

    ``supplied`` holds ``L`` in the global frame (RSPt's ``ml``/``mlp``); the reference is
    :func:`shell_operator_matrices`, which is ``L`` in the frame ``rot_to_spherical`` defines. On the shell's
    block, ``L_supplied_i = sum_j R_ij L_ref_j``; ``R`` is the least-squares solution in the Frobenius inner
    product (the Gram matrix of the reference components is not diagonal for a partial shell), then the
    nearest proper rotation. A shell with ``l = 0`` has no
    orbital moment and gets ``None``, as does every shell when ``supplied`` has no ``lz``/``l_plus``.

    Shells of one cluster may sit on different sites, so each gets its own frame.
    """
    lv = supplied.orbital_vector()
    if lv is None:
        return [None] * len(shell_ls)
    rot = np.asarray(rot_to_spherical, dtype=complex)
    out = []
    for l, mats, cols in zip(shell_ls, shell_operator_matrices(rot, shell_ls), _shell_slices(shell_ls)):
        if l == 0:
            out.append(None)
            continue
        lz, lp = mats[0], mats[1]
        ref = LocalOperators._cartesian(lz, lp)
        r_a = rot[:, cols]
        proj = r_a @ r_a.conj().T  # projector on the shell's (possibly partial) space
        block = np.stack([proj @ v @ proj for v in lv])
        gram = np.array([[np.real(np.vdot(ref[j], ref[k])) for k in range(3)] for j in range(3)])
        if np.linalg.cond(gram) > 1e10:  # a projected space too small to carry the vector (e.g. one orbital)
            out.append(None)
            continue
        overlap = np.array([[np.real(np.vdot(ref[j], block[i])) for j in range(3)] for i in range(3)])
        m = overlap @ np.linalg.inv(gram)
        r = _nearest_rotation(m)
        resid = np.tensordot(r, ref, axes=(1, 0)) - block
        out.append(SiteFrame(r, float(np.max(np.abs(resid)))))
    return out


def dipole_from_circular(dp, dm, dz):
    """Cartesian ``(3, n, n)`` dipole from RSPt's ``Dpm`` triple (inverse of ``dipole_circular``)."""
    dp, dm, dz = (np.asarray(x, dtype=complex) for x in (dp, dm, dz))
    s = np.sqrt(0.5)
    return np.stack([s * (dp + dm), s * (dp - dm) / 1j, dz])


__all__.append("dipole_from_circular")


_MATRIX_KEYS = ("sz", "s_plus", "lz", "l_plus", "s_axis")


def _encode(m):
    m = np.asarray(m, dtype=complex)
    return np.stack([m.real, m.imag], axis=-1).tolist()


def _decode(raw, what):
    arr = np.asarray(raw, dtype=float)
    if arr.ndim < 3 or arr.shape[-1] != 2:
        raise ValueError(f"{what} must be a nest of [re, im] pairs, got shape {arr.shape}")
    return arr[..., 0] + 1j * arr[..., 1]


def operators_to_header(ops):
    """JSON-ready ``operators`` header block of a :class:`LocalOperators` (matrices as ``[re, im]`` nests)."""
    block = {k: _encode(getattr(ops, k)) for k in _MATRIX_KEYS if getattr(ops, k) is not None}
    if ops.dipole is not None:
        block["dipole"] = _encode(ops.dipole)
    if ops.symmetry_ops:
        block["symmetry_ops"] = [{"matrix": _encode(u), "antiunitary": bool(a)} for u, a in ops.symmetry_ops]
    block["source"] = ops.source
    block["frame"] = ops.frame
    return block


def operators_from_header(block):
    """Inverse of :func:`operators_to_header`; ``None`` for a missing block."""
    if block is None:
        return None
    kwargs = {k: _decode(block[k], k) for k in _MATRIX_KEYS if k in block}
    if "dipole" in block:
        kwargs["dipole"] = _decode(block["dipole"], "dipole")
    kwargs["symmetry_ops"] = tuple(
        (_decode(item["matrix"], "symmetry_ops"), bool(item["antiunitary"])) for item in block.get("symmetry_ops", ())
    )
    return LocalOperators(source=block.get("source", "file"), frame=block.get("frame", "local"), **kwargs)


__all__ += ["operators_to_header", "operators_from_header"]
