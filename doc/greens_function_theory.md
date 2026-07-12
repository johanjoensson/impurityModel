# Green's functions in impurityModel: theory and practical guide

This document explains the physics and the algorithms behind every way this code can compute
an interacting Green's function, written for computational physicists rather than specialists
in numerical linear algebra. It covers the objects (the impurity model, the finite-temperature
Green's function, the variational many-body basis), the three resolvent engines
(`gf_method="lanczos"`, `"bicgstab"`, and the experimental spectrum slicing), and the two
production use cases — the self-energy for self-consistent DFT+DMFT and core-level
spectroscopy. Every performance or accuracy claim quoted here was *measured* on this code; the
raw tables live in the engineering logs under `doc/plans/` and are cited where relevant.
Numbered references `[N]` are collected in section 9.

Notation: $c^\dagger_i$ creates an electron in spin-orbital $i$; $z = \omega + i\delta$ is a
complex frequency ($\delta > 0$ retarded); $\beta = 1/\tau$ with $\tau = k_B T$ in the energy
units of the Hamiltonian (RSPt supplies Rydberg); $|m\rangle, E_m$ are many-body eigenstates.

---

## 1. The physical problem and its objects

### 1.1 The Anderson impurity model from DFT+DMFT

The solver diagonalizes a finite Anderson impurity model (AIM) [1]

$$
H \;=\; \underbrace{\sum_{ij} h^{\mathrm{corr}}_{ij}\, c^\dagger_i c_j}_{H_{\mathrm{imp}}}
\;+\; \underbrace{\tfrac12 \sum_{ijkl} U_{ijkl}\, c^\dagger_i c^\dagger_j c_l c_k}_{H_U}
\;+\; \underbrace{\sum_{\mu\nu} (h_{\mathrm{bath}})_{\mu\nu}\, c^\dagger_\mu c_\nu}_{H_{\mathrm{bath}}}
\;+\; \underbrace{\sum_{i\mu} \big( V_{\mu i}\, c^\dagger_\mu c_i + \mathrm{h.c.} \big)}_{H_{\mathrm{hyb}}},
$$

where $i,j$ run over the correlated impurity spin-orbitals (e.g. the ten $3d$ spin-orbitals)
and $\mu,\nu$ over a finite set of bath spin-orbitals. In the DFT+DMFT workflow [2,3] this
model is *derived*: RSPt hands over the one-body impurity block $h^{\mathrm{corr}}$ (in a
crystal-field basis, Rydberg units, with the double counting already subtracted), the Coulomb
tensor $U_{ijkl}$ in physicists' convention
($U4[i,j,k,l] = \langle ij|V|kl\rangle$, see `atomic_physics.getUop_from_rspt_u4` and
`selfenergy.get_Sigma_static`), and the continuous hybridization function on a mesh. The bath
parameters $(h_{\mathrm{bath}}, V)$ are then a **discrete fit** to that hybridization,

$$
\Delta(\omega) \;\approx\; V^\dagger \big[(\omega + i\delta)\,\mathbb{1} - h_{\mathrm{bath}}\big]^{-1} V ,
$$

(`selfenergy.hyb`; the fitted $h_{\mathrm{bath}}, V$ and the fit residual are archived in the
`Bath fit` group of `impurityModel_data.h5`). Everything downstream — eigenstates, Green's
functions, $\Sigma$, spectra — is exact diagonalization *of the fitted model*, in the spirit
of the ED impurity solver of Caffarel and Krauth [4]. Two consequences worth internalizing:

* the *quality* of $\Sigma$ against the lattice problem is bounded by the bath fit, not by the
  solver's convergence knobs, and
* the solver's non-interacting reference $G_0$ used in the Dyson equation
  (section 6) must be built from the **fitted**
  $\Delta$, not the input mesh data, so that solver and reference describe the same model.

The impurity one-body block may be rotated to a symmetry-adapted basis before solving
(`impurity_symmetry_rotation`): diagonalizing $h^{\mathrm{corr}}$ collapses the Green's-function
block structure to its finest form (e.g. ten $1{\times}1$ blocks for cubic $e_g/t_{2g}$), but
is applied only when it does not densify $U$ (spin-orbit-coupled $j,m_j$ bases would). All
outputs are rotated back to the caller's basis.

### 1.2 The finite-temperature Green's function

The central object is the finite-temperature, fixed-particle-number impurity Green's function
in Lehmann form. With a thermal ensemble of low-lying eigenstates
$\{|m\rangle, E_m\}$, Boltzmann weights $w_m = e^{-\beta(E_m - E_0)}$ and
$Z = \sum_m w_m$,

$$
G_{ij}(z) \;=\; \frac{1}{Z} \sum_m w_m \Big[
\underbrace{\langle m|\, c_i \,\big(z + E_m - H\big)^{-1} c^\dagger_j\, |m\rangle}_{\text{addition (IPS)}}
\;+\;
\underbrace{\langle m|\, c^\dagger_j \,\big(z - E_m + H\big)^{-1} c_i\, |m\rangle}_{\text{removal (PS)}}
\Big].
$$

Both terms are matrix elements of a **resolvent** $(z' - H)^{-1}$ between *seed states*
$c^\dagger_j|m\rangle$ (an electron added) or $c_i|m\rangle$ (an electron removed) — this is
the single mathematical task every method in this document performs, and the reason they can
share all of their surrounding machinery. In the code the two spectral sides are computed as
separate work units and combined as
$G = (G^{\mathrm{IPS}} - (G^{\mathrm{PS}})^{T})/Z$ with the removal side evaluated on the
negated mesh (`calc_thermally_averaged_G` and the assembly in `get_Greens_function`); the
per-eigenstate frequency frame is always $\omega_P = \omega + i\delta + E_m$
(`calc_G`, `_gf_signed_axes` — the one place this sign convention lives).

Two caveats that follow from the ensemble:

* **Fixed $N$, not grand canonical.** The ensemble is over eigenstates of fixed total particle
  number, so grand-canonical identities (the KMS detailed-balance relation between addition
  and removal spectral weight) hold only approximately; a diagnostic based on them is
  deliberately deferred (`gf_diagnostics`, module docstring).
* **Truncated ensemble.** Only `num_wanted` low-lying states are computed; if the highest
  retained state still carries Boltzmann weight, the thermal average is biased. This is
  detected (`check_thermal_weight_cutoff`) and drives an automatic retry with a doubled
  `num_wanted` in `calc_selfenergy`.

### 1.3 Block structure and symmetries

$G_{ij}$ is block-diagonal in any quantum number the Hamiltonian conserves. The blocks are
derived from the *hybridization-dressed* impurity matrix
$h^{\mathrm{corr}} + V^\dagger V$ (`impurity_block_structure`) — dressing matters, since two
impurity orbitals decoupled in $h^{\mathrm{corr}}$ may still couple through a shared bath.
Beyond block-diagonality, blocks may be related by symmetry: identical, transposed,
particle-hole conjugate, or particle-hole transposed
(`block_structure.BlockStructure`, applied in `build_full_greens_function`), and only one
representative per equivalence class is computed. For NiO without spin-orbit coupling the ten
$d$ spin-orbitals give ten $1{\times}1$ blocks in four equivalence classes; antiferromagnetic
NiO couples orbitals into blocks of width up to 4, making $G$ genuinely matrix-valued — the
reason every solver here is a *block* method.

### 1.4 The variational space: occupation restrictions

Exact diagonalization in the full Fock space of $\sim$60–150 spin-orbitals is impossible; the
many-body basis is a *selected* set of Slater determinants. The physical prior that makes
selection effective is that charge fluctuations away from the atomic-limit configuration are
progressively expensive. It is encoded as **occupation restrictions**: per-subset windows
$(N_{\min}, N_{\max})$ on the impurity shell, the valence baths (holes allowed) and the
conduction baths (electrons allowed), optionally per-$S_z$ *weighted* restrictions, and a
chain restriction for long discretized baths. One hard-won rule (the "NiO triplet lesson",
`basis_restrictions.build_excited_restrictions`): the impurity window must bound the
**whole impurity as one subset**, not each orbital-symmetry group separately — per-group
windows pin the $e_g/t_{2g}$ ratio and $S_z$ and can silently exclude the correct multiplet.

The restrictions define the space; the next section describes how the important determinants
*within* it are found.

---

## 2. Finding the thermal states

The Green's-function methods act on eigenstates $|m\rangle$ that must first be computed. This
section is self-contained theory for that stage; readers interested only in the resolvent
engines can skip to section 3.

### 2.1 CIPSI: building the basis

The ground-state basis is grown by **Configuration Interaction using a Perturbative Selection
made Iteratively** (CIPSI) [5,6]. Starting from a small reference (seeded by a cheap
memory-bounded unrestricted Hartree–Fock calculation that also fixes the initial occupation
sector, `hartree_fock.hartree_fock_seed_occupation`), the loop alternates:

1. **Diagonalize** $H$ in the current variational space $\mathcal{V}$, giving
   $|\Psi\rangle = \sum_{D \in \mathcal{V}} a_D |D\rangle$ and energy $E$.
2. **Select**: for every determinant $|\mu\rangle \notin \mathcal{V}$ connected to $\Psi$ by
   $H$, estimate its second-order Epstein–Nesbet [7] energy contribution

   $$
   \delta E^{(2)}_\mu \;=\; \frac{\big|\langle \mu | H | \Psi \rangle\big|^2}
                                  {\big|E - \langle \mu|H|\mu\rangle\big|} ,
   $$

   and admit those with $\delta E^{(2)}_\mu \ge$ `de2_min` (`cipsi_solver._calc_de2`; the
   denominator is used in magnitude — a signed near-zero denominator would otherwise promote
   irrelevant determinants).
3. Repeat until the energy is converged and no candidate passes the threshold.

Under a determinant budget (`truncation_threshold`), selection switches to a **fixed-budget
refinement**: candidates are ranked by $\delta E^{(2)}$ and admitted through a collective
amplitude bisection until the budget is filled, followed by top-$K$ amplitude truncation of
the converged state. Measured on the test workloads, the ground-state energy remains exact to
$\sim 10^{-14}$ even when the cap binds — the discarded determinants carry the amplitude tail,
not the energy. The MPI layout is one hash-owner per determinant
(`hash(determinant) % n_{\mathrm{ranks}}`), and *all* selection decisions derive from
globally reduced quantities, so the basis trajectory is rank-count independent up to
summation-order rounding at exact ties.

### 2.2 Block-Lanczos eigensolvers: TRLM and IRLM

Within the selected basis (dimension up to $\sim 10^6$), the lowest eigenpairs are found by
block-Krylov iteration. The Lanczos idea in one sentence: applying $H$ repeatedly to a block
of $p$ start vectors builds the Krylov space
$\mathcal{K}_k = \mathrm{span}\{Q_1, HQ_1, \dots, H^{k-1}Q_1\}$, in which $H$ is represented
by a small block-tridiagonal matrix

$$
T_k \;=\; \begin{pmatrix}
\alpha_1 & \beta_1^\dagger & & \\
\beta_1 & \alpha_2 & \beta_2^\dagger & \\
 & \beta_2 & \ddots & \ddots \\
 & & \ddots & \alpha_k
\end{pmatrix},
\qquad H Q_j = Q_{j-1}\beta_{j-1}^\dagger + Q_j \alpha_j + Q_{j+1} \beta_j ,
$$

whose eigenpairs (Ritz pairs) converge to extremal eigenpairs of $H$ exponentially fast in
$k$ [8,9]. Blocks ($p > 1$) resolve degenerate multiplets — essential here, where crystal-field
and spin degeneracies are the norm.

Unbounded $k$ is not affordable (each $Q_j$ is a distributed many-body state), so the
iteration is **restarted**, keeping the best part of the space:

* **TRLM** (thick-restart Lanczos [10]): after $k$ steps, keep the $\ell$ best Ritz vectors,
  compress them to the front of the basis, and continue the recurrence behind them. The
  retained Ritz block must be numerically orthonormal for the shortcut to be valid — the code
  gates this on an explicit tolerance and falls back to a Rayleigh–Ritz rebuild when it fails.
* **IRLM** (implicitly-restarted block Lanczos with locking [11,12]): converged Ritz vectors
  are *locked* (frozen and deflated out of the active recurrence), and the restart applies a
  polynomial filter implicitly via the EA16 purge/restart of Baglama, Calvetti and Reichel
  [12]. Locked directions must also be deflated from the *inner sweep*, or orthogonality to
  them regrows and the "converged" energy can drop below the exact minimum — a measured
  failure mode, fixed by construction in this code.

Both restart flavors share one deflation routine (`_cholesky_or_deflate`) built on a principle
that recurs throughout this document: **rank and breakdown are different questions at
different scales**. Whether a block's columns are linearly dependent (deflate the direction)
is judged *relative* to the block's own largest singular value; whether the block is
numerically zero (the Krylov space closed — an invariant subspace) is an *absolute* question
that needs an external scale, here $\sim\|H\|$. Fusing the two (the historical
`evals > \epsilon \cdot \max(\lambda_{\max}, 1)`) silently returned warm starts unrefined — the
class of bug that motivated the separation.

### 2.3 Orthogonality loss and reorthogonalization

In exact arithmetic the Lanczos vectors are orthonormal; in floating point they lose
orthogonality *precisely when a Ritz pair converges* (Paige's theorem [13]): the rounding
errors align with converged directions, which then re-enter the recurrence as duplicated
"ghost" copies. The available cures, in decreasing cost (`Reort` modes, shared by the
eigensolvers and the Green's-function recurrences):

* **FULL** — orthogonalize every new block against every stored one, every step. Exact, and
  requires *retaining all Krylov vectors* — the memory term to remember for
  section 3.
* **PARTIAL** — track a running estimate $\omega_{jk}$ of the worst inner product between the
  new and stored blocks via Simon's recurrence [14], and reorthogonalize only when it crosses
  $\sqrt{\varepsilon}$. Near-FULL accuracy at a fraction of the projections. Two hard lessons
  are baked into this implementation: the estimator's seed needs an $\|H\|$ proxy for warm
  starts (or it never fires), and after a reorthogonalization the estimator must be reset to
  the *measured* residual overlaps, not optimistically to $\varepsilon$ — an optimistic reset
  blinded the estimator and diverged a production Green's function at iteration 833.
* **SELECTIVE** — orthogonalize only against *converged* Ritz vectors (Parlett–Scott [15]),
  gated on the Ritz pair's own convergence weight.
* **PERIODIC** — reorthogonalize on a fixed cadence; simple, no estimator to mistrust.
* **NONE** — let orthogonality go. For *eigenvalues* this is dangerous (ghosts); for
  *broadened resolvents* it is remarkably safe, as quantified below.

### 2.4 The thermal ensemble

The eigensolver returns the `num_wanted` lowest states; those within
$\Delta E_{\mathrm{cut}} = -\tau \ln(10^{-4})$ of the ground state enter the Boltzmann
ensemble. `gf_diagnostics.check_thermal_weight_cutoff` flags a truncated ensemble (highest
retained state still carrying weight), which triggers the automatic `num_wanted` doubling in
`calc_selfenergy`. Degeneracies are common (spin multiplets), so interpreting per-state
observables requires the whole multiplet — one reason the observable reports aggregate
Casimirs ($S^2, L^2, J^2$) over whole shells.

---

## 3. Method I — the block-Lanczos continued fraction (`gf_method="lanczos"`)

### 3.1 The Haydock recursion

The oldest and still default route to a resolvent matrix element [16,17,18]: run the Lanczos
recurrence of section 2.2 **seeded by the physical state itself**,
$Q_1 = \mathrm{qr}(\,[c^\dagger_j|m\rangle]_j\,)$ (the thin-QR factor $r$ of the seed block is
kept, `build_qr`). Because $T_k$ is the representation of $H$ in the Krylov space of the seed,
the resolvent matrix element over the seed block is, exactly,

$$
G(z) \;=\; r^\dagger \big(\omega_P\,\mathbb{1} - T_k\big)^{-1}\, r \Big|_{\text{top-left block}},
\qquad \omega_P = \omega + i\delta + E_m ,
$$

evaluated in the code by the numerically stable block continued fraction
(`calc_G`, `_block_cf_inverse`):

$$
G(z) = r^\dagger \cfrac{1}{\omega_P - \alpha_1 -
        \beta_1^\dagger \cfrac{1}{\omega_P - \alpha_2 -
        \beta_2^\dagger \cfrac{1}{\omega_P - \alpha_3 - \cdots}\,\beta_2}\,\beta_1}\; r .
$$

The essential property: $T_k$ is **frequency-independent**. One recurrence of $m$ blocks
yields $G$ on *every* frequency of *both* axes — this is the shift invariance of Krylov
spaces, $\mathcal{K}(z - H, v) = \mathcal{K}(H, v)$, and it is the deep reason the continued
fraction is the default method (see also section 4.3). Physically, $k$ blocks resolve the
spectral function through its first $2k$ moments; broadening $\delta$ smears what the
truncated fraction has not resolved.

### 3.2 What orthogonality loss does to a resolvent

Ghost Ritz values *duplicate* spectral weight, they do not misplace it (Cullum–Willoughby
[19]); under a Lorentzian broadening the duplication largely cancels in the continued
fraction. This is why `reort="none"` — which retains **no Krylov store at all** — is the
production self-energy default, and why it is measured to be not merely acceptable but
excellent: on the NiO 1-bath real-axis workload, `none` agrees with `full` to
$1.6\times10^{-10}$ while `partial` sits at $2.5\times10^{-8}$, at a quarter of the wall time
(`doc/plans/bicgstab_per_frequency_gf.md`, Phase 3a-quinquies). The cost of the reorthogonalized
modes is concrete: retaining $m$ blocks of width $p$ over $C$ determinants costs
$16\,m\,p\,C$ bytes (complex128), which on production runs is the difference between the
1.4 GiB (`none`) and 2.0 GiB (`partial`) peaks measured on FCC Ni.

### 3.3 Stopping: the mesh-aware convergence monitor

The recurrence stops when $G$ — evaluated **on the frequencies the caller will actually use**
— stops changing: the monitor (`_make_gf_convergence_monitor`, fed by `_gf_eval_meshes`)
compares the resolvent after $k$ and $k-1$ blocks on a subsample of the caller's meshes, per
axis, and requires the relative change to stay below $\max(\texttt{slaterWeightMin}^2, 10^{-9})$
for two consecutive blocks. Converging "where you evaluate" matters enormously on the
Matsubara axis, whose points $i\omega_n$ sit at least $\sqrt{E_k^2 + \omega_n^2}$ from every
pole: a Matsubara-only self-energy needs $\sim$10x fewer blocks than the real-axis resolvent
the historical band-wide monitor insisted on resolving (measured: 360 vs 3496 blocks on NiO,
$\sigma$ agreement $5\times10^{-13}$).

### 3.4 Basis truncation: exactness on the retained subspace

The recurrence's matvec *discovers* new determinants every step (the live support), and on
open systems that support grows without bound — the measured FCC Ni memory wall
(11.7 GiB and climbing, uncapped). The cap (`truncation_threshold`, enforced by
`_CappedBasisProxy`) freezes growth at $C$ determinants with amplitude-ranked admission at the
overflow step. The theoretical statement that makes this safe: after the freeze the recurrence
is an **exact block Lanczos of the projected operator $PHP$** ($P$ = projector on the retained
set). Sketch: every previously accepted Krylov block lies inside $P$, so
$\langle Q_j | P w \rangle = \langle Q_j | w \rangle$ — the projector is invisible to all
orthogonalizations, the recurrence coefficients are those of $PHP$, and the continued fraction
is the causal, exact resolvent of a Hermitian (projected) operator. Moments up to the freeze
step are exact with respect to the full $H$.

The projection is *not* harmless for derived quantities: a causal $G_P$ does **not** make
$\Sigma = G_0^{-1} - G_P^{-1}$ causal, because the missing spectral weight enters through the
inverse. Measured on FCC Ni at $C = 4\times10^5$: every method's $\Sigma(i\omega_n)$ —
including per-frequency solves converged to $10^{-8}$ — violates
$\mathrm{Im}\,\Sigma_{ii} \le 0$ and is rejected by the physicality guard. When the cap binds
this hard, the resolution is more retained weight (larger cap, more ranks) or a better
subspace, not a different solver.

### 3.5 Work distribution

All Green's-function drivers share one distribution engine: the flat work units are
(block $\times$ spectral side $\times$ eigenstate-chunk), load-balanced by a seed-mass cost
model and executed after a **single** communicator split (`enumerate_gf_units`,
`unit_cost_weights`, `run_units_distributed`). Two granularity knobs move work between shared
Krylov spaces and independent units:

* **eigenstate grouping** (`GF_EIGENSTATE_GROUP`): stack $g$ eigenstates' seeds into one
  width-$g p$ recurrence — the shared $T_k$ serves every stacked state (each keeps its own
  columns of $r$ and its own $E_m$ shift), trading matvec sharing against wider-block
  reorthogonalization;
* **operator splitting** (`GF_OPERATOR_SPLIT`): compute an $n{\times}n$ block from scalar
  recurrences only, using the polarization identity
  $G_{ij} = \tfrac12\big[S(v_i{+}v_j) - i\,S(v_i{+}iv_j) - (1{-}i)(G_{ii}{+}G_{jj})\big]$
  with $S(w) = \langle w|(z-H)^{-1}|w\rangle$ (`calc_G_pairwise`) — maximal communication-free
  parallelism at the price of redundant Krylov building.

---

## 4. Method II — per-frequency Krylov solves (`gf_method="bicgstab"`)

### 4.1 The resolvent as a linear system

Instead of one recurrence for all frequencies, solve at each mesh point $z$

$$
\big(z + E_m - H\big)\, X_j \;=\; c^\dagger_j|m\rangle , \qquad
G_{ij}(z) = \langle c^\dagger_i m \,|\, X_j \rangle ,
$$

(`block_Green_bicgstab`). The solver state is a fixed handful of block vectors, there is no
Krylov store and *no orthogonality to lose* — accuracy is set by the residual tolerance alone.
The driver additionally **rebuilds the many-body basis from the current seed + warm-start
support at every point and discards it afterwards**, so its footprint is the largest
single-point support, not the union over the mesh; a finite `truncation_threshold` caps even
that per point (fresh `_CappedBasisProxy`; post-freeze the solve is exact on $PHP$, same
contract and same oracle test as the capped recurrence).

The honest measured verdict (`doc/plans/bicgstab_per_frequency_gf.md`, Phases 3a–3b): this is
**not a memory win on the workloads measured**. On FCC Ni the per-point support *equals* the
union support — the capped ground state already saturates the budget, its seeds inherit that
support, and a Matsubara point's solution mixes every pole. What the method delivers instead
is *reliability on frozen subspaces* (every solve converged to $10^{-8}$ where the capped
recurrence's monitor stalled at $10^{-4}$) and embarrassing frequency parallelism, at
$\sim$5x end-to-end wall. On the Matsubara axis it reproduces $\Sigma$ to $2.4\times10^{-8}$
relative with a bit-identical $\Sigma^{\mathrm{static}}$.

### 4.2 BiCGSTAB, warm starts, and near-pole stagnation

BiCGSTAB [20] is a transpose-free Krylov solver: two matvecs per iteration, no stored basis.
Its convergence rests on maintaining bi-orthogonality against a fixed *shadow residual*
$\tilde r_0$; when the true residual loses overlap with $\tilde r_0$ — which happens for
nearly singular, indefinite systems, i.e. real-axis points within $\sim\delta$ of a pole —
the iteration stagnates without failing. Three layers of the driver address this:

* **Warm starts.** $G(z)$ is locally rational, so the solution at the next mesh point is
  predicted by quadratic Lagrange extrapolation in $z$ through the last three solutions
  (the measured optimum; cubic amplifies solver noise). Crucially, the tolerance contract
  makes warm starts pay in *iterations*, not in silently tighter targets: `atol` is relative
  to the right-hand side norm, enforced by deflating the *normalized* residual Gram and
  rescaling (the entry logic of `block_bicgstab`). Warm-started Matsubara solves cost
  $\sim$3 iterations against a cold start's $\sim$12.
* **Restarts.** An unconverged solve re-enters the solver with its current iterate: the
  residual block is re-deflated and a *fresh* shadow residual chosen — the standard cure,
  progress-gated so a truly stuck point stops early. Measured on the SIAM-6 anchor:
  near-pole error $6.6\times10^{-3} \to 3.9\times10^{-9}$ at unchanged cost.
* **Rank-deficient seeds** (a filled orbital annihilated by $c^\dagger$, symmetry-degenerate
  columns) are handled by deflating the seed block to its independent directions and
  reconstructing dependent columns by linearity — never by solving a singular block system.

### 4.3 The GMRES fallback

For the points that stagnate through all restarts (measured: 25 of 1440 real-axis solves on
NiO at $\delta = 0.01$, residual $\sim 2\times10^{-2}$), the driver falls back to restarted
block **GMRES** [21] (`gmres.block_gmres`, block Arnoldi): the residual is *minimized* over
the Krylov space at every step — monotone by construction, no shadow residual, no breakdown
modes — at the price of storing one Krylov block per iteration (bounded by the restart length,
`GF_GMRES_RESTART = 40`). The fallback is warm-started from BiCGSTAB's partial iterate and
runs *before* the solution enters the warm-start history, so a rescued point also repairs the
extrapolation chain its stagnated result would have poisoned. Measured: 13 of 16 flagged NiO
points fully rescued (stragglers at $2.9\times10^{-8}$), $\sigma^{\mathrm{real}}$ error
$2\times10^{-4} \to 5.4\times10^{-7}$ — *below* the Lanczos partial-vs-full spread — for
$\sim$2% extra iterations.

A theoretical remark that explains the pecking order of this whole document: for the shifted
**Hermitian** system $(z-H)X = v$, the optimal-residual method can be built on the Hermitian
Lanczos three-term recurrence (shifted MINRES / shifted Krylov methods [22]), and if only the
matrix elements $\langle v'|X\rangle$ are wanted — never $X$ itself — that construction *is*
the continued fraction of section 3, evaluated at every shift simultaneously. Per-frequency
solvers can therefore never beat the recurrence at its own game (all frequencies, one basis);
their niche is everything the recurrence cannot do: per-point tolerance certificates, warm
starts, capped-subspace reliability, and resolvents whose right-hand side *changes with
frequency* — which is exactly the RIXS intermediate-state problem
(section 7.2).

### 4.4 When to use which (methods I and II)

| situation | method | why |
|---|---|---|
| Matsubara-only self-energy, healthy basis | `lanczos`, `reort="none"` | one recurrence, no store, measured excellent |
| real-axis spectra, healthy basis | `lanczos`, `reort="none"` (or `partial` for belt-and-braces) | same; ghosts smeared by $\delta$ |
| cap-frozen basis where the CF monitor cannot converge | `bicgstab` | per-point residual certificates; measured all-converged where Lanczos stalled |
| real axis at very small $\delta$ with strict per-point accuracy | `bicgstab` (+ automatic GMRES fallback) | stagnation handled and *reported* |
| far more ranks than matvec parallelism can absorb | `bicgstab` | frequency axis is embarrassingly parallel |
| frequency-dependent right-hand side (RIXS $R_1$) | `block_bicgstab` directly | no shared Krylov space exists to exploit |

---

## 5. Method III (outlook) — spectrum slicing with Chebyshev filters (`gf_method="sliced"`, experimental)

*Status: under construction on this branch; this section records the theory and will be
updated with the measured verdict.*

All methods above pay the full **live determinant support** of the seed's Krylov space — the
term that actually exhausts memory (section 3.4). The one idea that attacks it directly is
energy localization *by construction*. Choose a Chebyshev-polynomial partition of unity on
the spectral interval of $H$: window functions $p_s(H)$, built as differences of smoothed step
functions so that $\sum_s p_s \equiv 1$ identically (Jackson-damped kernel-polynomial
construction [23]). Then, exactly,

$$
G_{ij}(z) \;=\; \sum_s \big\langle v_i \big| (z-H)^{-1} \underbrace{p_s(H)\, v_j}_{v_j^s} \big\rangle ,
$$

where each filtered seed $v^s = p_s(H)v$ carries spectral weight only inside slice $s$ (plus
controlled leakage). Each slice term is an independent work unit with its *own, discarded*
basis, solved by the per-frequency driver of section 4 with an unfiltered bra (its natural
cross-element form); the filter itself is a three-term Chebyshev recurrence
($t_{n+1} = 2\tilde H t_n - t_{n-1}$, three live vectors, one pass serving all slices), with
spectral bounds from a short Lanczos run. The bet under measurement: whether the
$\varepsilon$-support of $p_s(H)v$ at production amplitude cutoffs is genuinely smaller than
the union support — for a metal's mid-band many-body states, delocalization in Fock space is
the honest risk, and the calibration probe measures exactly this before the driver's memory
model is trusted.

---

## 6. Use case A — the DMFT self-energy (`calc_selfenergy`)

### 6.1 Dyson equation on the fitted model

With the interacting $G$ from any method above, the impurity self-energy per Green's-function
block is (`selfenergy.get_sigma`)

$$
\Sigma(z) \;=\; G_0^{-1}(z) - G^{-1}(z), \qquad
G_0^{-1}(z) \;=\; z\,\mathbb{1} - h^{\mathrm{corr}} - \Delta(z),
$$

with the hybridization $\Delta(z) = V^\dagger (z - h_{\mathrm{bath}})^{-1} V$ built from the
**fitted** bath (section 1.1) so that $G_0$ and $G$ describe the same finite model — using the
input-mesh hybridization here would fold the bath-fit residual into $\Sigma$. The static
(Hartree–Fock) part, used for the high-frequency limit and by RSPt's double-counting
bookkeeping, is evaluated directly from the thermal impurity density matrix
(`get_Sigma_static`):

$$
\Sigma^{\mathrm{static}} \;=\; \sum_{ij} \big( U4[j, :, i, :] - U4[j, :, :, i] \big)\, \rho_{ij} .
$$

Both $\Sigma(i\omega_n)$ (for the DMFT cycle) and $\Sigma(\omega + i\delta)$ (for analysis)
come from the same machinery, differing only in the mesh handed to the Green's-function
driver. Two error-propagation facts to keep in mind:

* **The inversion amplifies where $|G|$ is small.** $\delta\Sigma \sim G^{-1}\,\delta G\,G^{-1}$,
  so between spectral features (small $|G|$) a Green's-function error is magnified — the
  measured reason the per-frequency driver's default residual tolerance ($10^{-8}$) had to be
  tightened (and its stagnating points GMRES-rescued) before $\sigma^{\mathrm{real}}$ at
  $\delta = 10^{-2}$ matched the recurrence.
* **Truncation is not causal in $\Sigma$** (section 3.4): the physicality guard
  (`check_greens_function` / the $\mathrm{Im}\,\Sigma_{ii} \le 0$ check) is the last line of
  defense and *raising* is collective (`_raise_together`) so no MPI rank deadlocks.

When RSPt requests it, the solver also *determines* the double counting: shift the impurity
levels until either a chosen spectral peak sits at a target energy (`fixed_peak_dc`) or the
thermal impurity occupation matches a target (`fixed_occupation_dc`).

### 6.2 Practical recipe

Production defaults (what the RSPt interface records in the archives): `gf_method="lanczos"`,
`reort="none"`, `sparse_green=True`, `truncation_threshold=None` (auto-sized from per-rank
RAM by the memory model below), `dN=None`. The diagnostics report printed at the end of every
run is the honest summary — read it: `thermal_cut` (ensemble truncation → auto-retry),
`basis_cap` (cap hit; result exact on the retained subspace), `lanczos`/`bicgstab` (solver
convergence, GMRES rescues), `mesh_density` (quadrature quality at the given $\delta$),
`causality` (hard failure). The memory model
(`memory_estimate.estimate_gf_peak_bytes`) prices a run as

$$
\mathrm{peak} \;\approx\; C\,\big( s_{\mathrm{live}} + 16\,m\,p \big) \quad\text{per rank},
$$

$C$ = determinant cap, $s_{\mathrm{live}} \approx 450\text{–}550$ B/determinant (basis
bookkeeping + the recurrence's three live blocks), and the Krylov-store term only for
`reort != "none"`; the per-frequency driver replaces $16\,m\,p$ by its $\sim$12 live blocks
plus the GMRES fallback transient. `suggest_truncation_threshold` inverts this against
available RAM (cgroup-aware), and `max_colors_within_budget` caps how many work units run
concurrently.

## 7. Use case B — core-level spectroscopy (`get_spectra` / `spectra.py`)

### 7.1 One resolvent, many spectroscopies

Fermi's golden rule for any transition operator $T$ is the same resolvent matrix element yet
again:

$$
I(\omega) \;=\; -\frac{1}{\pi}\, \mathrm{Im} \sum_m \frac{w_m}{Z}\,
\big\langle m \big| T^\dagger \big(\omega + i\Gamma + E_m - H\big)^{-1} T \big| m \big\rangle ,
$$

with the broadening $\Gamma$ now playing a physical role: the inverse core-hole lifetime.
The spectroscopies differ only in $T$:

* **PES / IPS** (photoemission / inverse photoemission): $T = c_i$ / $c^\dagger_i$ over the
  valence orbitals (`getPhotoEmissionOperators`, `getInversePhotoEmissionOperators`) — these
  are literally the removal/addition Green's functions of section 1.2, traced.
* **XAS** (x-ray absorption): the dipole operator between the core and valence shells,
  $T_\varepsilon = \sum_\alpha \varepsilon_\alpha T_\alpha$ with Gaunt-coefficient matrix
  elements (`getDipoleOperator`; conventions after Eder's multiplet notes). The final states
  contain a core hole, so the excited-sector restrictions differ from the ground state's —
  the `imp_change`/`val_change`/`con_change` windows passed to the spectra drivers encode
  exactly which charge rearrangements the intermediate space may explore.
* **XPS**: the sudden removal of a core electron (a scalar seed in the core-hole sector).
* **NIXS** (non-resonant inelastic x-ray scattering): $T = e^{i q\cdot r}$ expanded in
  spherical harmonics with radial integrals $\langle R_i | j_k(qr) | R_j \rangle$
  (`getNIXSOperator`) — access to multipole transitions beyond dipole at large $|q|$.

Since transition operators are *linear* in the polarization vector, an $n$-polarization
measurement is a contraction of one Cartesian component tensor: `getSpectra_tensor` runs a
single block recurrence over the component seeds and reads off every polarization (and
circular/arbitrary combinations) afterwards — with symmetry-equivalent seed redundancy removed
automatically by the rank deflation of the block QR, which is the *correct* dedup (an
XAS-style group rule does not generalize to tensors).

### 7.2 RIXS: the Kramers–Heisenberg amplitude

Resonant inelastic x-ray scattering is a two-photon process through a core-hole intermediate
state [24,25]; the Kramers–Heisenberg amplitude requires **two nested resolvents**
(`getRIXSmap_tensor`, `_rixs_map_flat`):

$$
C_{\alpha\alpha'\beta\beta'}(\omega_{\mathrm{in}}, \omega_{\mathrm{loss}}) =
\big\langle \psi^{(2)}_{\alpha\beta} \big|\, R_2(\omega_{\mathrm{loss}})\, \big| \psi^{(2)}_{\alpha'\beta'} \big\rangle,
\qquad
\psi^{(2)}_{\alpha\beta} = T^{\mathrm{out}}_\beta\, R_1(\omega_{\mathrm{in}})\, T^{\mathrm{in}}_\alpha |g\rangle ,
$$

with $R_{1,2} = (\omega + i\Gamma_{1,2} + E_g - H)^{-1}$; $\Gamma_1$ is the core-hole
(intermediate) inverse lifetime, $\Gamma_2$ the final-state broadening. The structure dictates
the solvers: $R_1$'s right-hand side is fixed but must be evaluated at **every incoming
frequency** — a genuinely per-frequency problem, solved by warm-started `block_bicgstab`
sweeping the $\omega_{\mathrm{in}}$ chunk on a per-point rebuilt basis (this driver is where
the per-frequency machinery of section 4 was born); $R_2$ over the loss axis has a fixed seed
block and is one block-Lanczos continued fraction per $\omega_{\mathrm{in}}$. The intermediate
space is confined by conserved-charge sector restrictions (the core hole shifts the sector,
`transition_sector_restrictions`), and the full rank-4 polarization tensor comes from one
solve per $\omega_{\mathrm{in}}$ regardless of how many polarization pairs are requested.

## 8. Decision guide and knob reference

**Method choice** — the table of section 4.4, plus: spectra drivers (`get_spectra`) currently
run Method I internally (their result contract is continued-fraction coefficients); the
self-energy path accepts `--gf_method {lanczos,bicgstab}`.

**The knobs that matter, and what they mean physically:**

| knob | meaning | guidance |
|---|---|---|
| `reort` | ghost suppression vs Krylov store (§2.3, §3.2) | `none` for resolvents (default); `partial` when eigen-accuracy of the recurrence itself matters |
| `truncation_threshold` | determinant cap = the memory budget (§3.4) | `None` → auto-size from RAM; watch `basis_cap` + `causality` diagnostics |
| `dN` (+ `imp/val/con_change`) | charge-fluctuation window of the excited sector (§1.4) | physical prior; measured **ineffective on metals** (FCC Ni: no reduction, 12% slower) |
| `slaterWeightMin` | amplitude cutoff on states/seeds | sets the accuracy floor $\sim$`slaterWeightMin`$^2$ of the CF monitor |
| `delta` | broadening: resolution on the real axis, core-hole lifetime in spectra | smaller δ = harder solves near poles (§4.2); mesh must resolve it (`check_mesh_density`) |
| `GF_BICGSTAB_ATOL` / `MAX_ITER` / `RESTARTS` | per-point solve contract (§4.2) | default `1e-8`; tighten to `1e-10` for real-axis Σ at small δ |
| `GF_GMRES_RESTART` / `MAX_RESTARTS` | fallback Arnoldi depth (§4.3) | 40 default; bounds the fallback's memory transient |
| `GF_EIGENSTATE_GROUP` / `GF_OPERATOR_SPLIT` | unit granularity (§3.5) | grouping shares matvecs; splitting maximizes independent units |
| `num_wanted`, `tau` | thermal ensemble (§2.4) | auto-retry doubles `num_wanted` on the truncation diagnostic |

## 9. References

1. P. W. Anderson, *Localized Magnetic States in Metals*, Phys. Rev. **124**, 41 (1961).
2. A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg, *Dynamical mean-field theory of
   strongly correlated fermion systems*, Rev. Mod. Phys. **68**, 13 (1996).
3. G. Kotliar et al., *Electronic structure calculations with dynamical mean-field theory*,
   Rev. Mod. Phys. **78**, 865 (2006).
4. M. Caffarel and W. Krauth, *Exact diagonalization approach to correlated fermions in
   infinite dimensions*, Phys. Rev. Lett. **72**, 1545 (1994).
5. B. Huron, J. P. Malrieu, P. Rancurel, *Iterative perturbation calculations of ground and
   excited state energies from multiconfigurational zeroth-order wavefunctions*,
   J. Chem. Phys. **58**, 5745 (1973).
6. Y. Garniron et al., *Selected configuration interaction dressed by perturbation*,
   J. Chem. Phys. **149**, 064103 (2018).
7. P. S. Epstein, Phys. Rev. **28**, 695 (1926); R. K. Nesbet, Proc. R. Soc. A **230**, 312 (1955).
8. C. Lanczos, *An iteration method for the solution of the eigenvalue problem of linear
   differential and integral operators*, J. Res. Natl. Bur. Stand. **45**, 255 (1950).
9. G. H. Golub and R. Underwood, *The block Lanczos method for computing eigenvalues*, in
   *Mathematical Software III* (1977).
10. K. Wu and H. Simon, *Thick-restart Lanczos method for large symmetric eigenvalue
    problems*, SIAM J. Matrix Anal. Appl. **22**, 602 (2000).
11. D. C. Sorensen, *Implicit application of polynomial filters in a k-step Arnoldi method*,
    SIAM J. Matrix Anal. Appl. **13**, 357 (1992).
12. J. Baglama, D. Calvetti, L. Reichel, *IRBL: an implicitly restarted block-Lanczos method*
    (EA16), SIAM J. Sci. Comput. **24**, 1650 (2003).
13. C. C. Paige, *The computation of eigenvalues and eigenvectors of very large sparse
    matrices*, PhD thesis, University of London (1971).
14. H. D. Simon, *The Lanczos algorithm with partial reorthogonalization*,
    Math. Comp. **42**, 115 (1984).
15. B. N. Parlett and D. S. Scott, *The Lanczos algorithm with selective orthogonalization*,
    Math. Comp. **33**, 217 (1979).
16. R. Haydock, V. Heine, M. J. Kelly, *Electronic structure based on the local atomic
    environment for tight-binding bands*, J. Phys. C **5**, 2845 (1972); **8**, 2591 (1975).
17. E. Dagotto, *Correlated electrons in high-temperature superconductors*,
    Rev. Mod. Phys. **66**, 763 (1994).
18. E. Koch, *The Lanczos Method*, in *The LDA+DMFT approach to strongly correlated
    materials*, Jülich Autumn School lecture notes (2011).
19. J. K. Cullum and R. A. Willoughby, *Lanczos Algorithms for Large Symmetric Eigenvalue
    Computations* (Birkhäuser, 1985).
20. H. A. van der Vorst, *Bi-CGSTAB: a fast and smoothly converging variant of Bi-CG*,
    SIAM J. Sci. Stat. Comput. **13**, 631 (1992).
21. Y. Saad and M. H. Schultz, *GMRES: a generalized minimal residual algorithm*,
    SIAM J. Sci. Stat. Comput. **7**, 856 (1986).
22. A. Frommer, *BiCGStab(l) for families of shifted linear systems*,
    Computing **70**, 87 (2003).
23. A. Weiße, G. Wellein, A. Alvermann, H. Fehske, *The kernel polynomial method*,
    Rev. Mod. Phys. **78**, 275 (2006).
24. H. A. Kramers and W. Heisenberg, *Über die Streuung von Strahlung durch Atome*,
    Z. Phys. **31**, 681 (1925).
25. L. J. P. Ament et al., *Resonant inelastic x-ray scattering studies of elementary
    excitations*, Rev. Mod. Phys. **83**, 705 (2011).
26. F. de Groot and A. Kotani, *Core Level Spectroscopy of Solids* (CRC Press, 2008).
27. M. W. Haverkort, M. Zwierzycki, O. K. Andersen, *Multiplet ligand-field theory using
    Wannier orbitals*, Phys. Rev. B **85**, 165113 (2012).
