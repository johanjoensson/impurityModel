# Interpreting the ground-state report

When `calc_gs` (`impurityModel.ed.groundstate`) finishes, rank 0 prints a report of the
low-energy eigenstates: expectation values, configuration weights, participation/entropy
measures, natural-orbital occupations and density matrices. This page walks through that
report in print order and explains, for each number, what it is, where it comes from in
the code, and what it can be used for.

The same information is saved machine-readable to `ground_state_statistics.json`
(`gs_statistics.save_gs_statistics`; path set by the `stats_path` argument of
`calc_gs`): the statistics dict plus a `truncation` record, the `entanglement` results,
and an `observables` dict bundling the magnetic/multiplet summary, the per-state
summary rows, the correlation and screening diagnostics and the energy decomposition
(complex values are stored as `[re, im]` pairs).

## The thermal ground state

Almost everything labelled *thermal* in the report is a Boltzmann average over the
low-energy eigenstates $|n\rangle$ with energies $E_n$:

$$
\langle \hat O \rangle = \sum_n w_n \langle n|\hat O|n\rangle,
\qquad
w_n = \frac{e^{-(E_n - E_0)/\tau}}{\sum_m e^{-(E_m - E_0)/\tau}} .
$$

- $\tau$ is the `tau` entry of the basis setup, in the same energy units as the
  Hamiltonian (eV in the standard workloads). Physically $\tau = k_B T$.
- The solver keeps at most `num_wanted` states (default 10) and only those within
  $E - E_0 \le -\tau \ln 10^{-4} \approx 9.2\,\tau$, i.e. states whose relative Boltzmann
  weight would be below $10^{-4}$ are discarded up front.
- Averaging matters even at $\tau \to 0$: a degenerate ground manifold (e.g. a spin
  triplet) enters with equal weights, so the thermal quantities are basis-independent
  manifold averages rather than properties of one arbitrary member of the manifold.
  This is the reason many per-state numbers are *not* individually meaningful while
  their thermal averages are — see [the per-eigenstate table](gs-report-per-eigenstate).

The averages are evaluated by `average.thermal_average_scale_indep` (density matrices,
energies) and `observables.thermal_observable_value` (Casimirs, spin correlations);
`gs_statistics._boltzmann_weights` computes the same $w_n$ for the statistics block.

## Before the report: occupation search and basis

These lines are printed while the ground-state sector and variational basis are being
determined (`groundstate.find_ground_state_basis`), before any observables.

```text
HF-seeded ground state occupation: {0: 2, 1: 6} ~ -53.888
Ground state occupation
 0 :   2
 1 :   6
E$_{GS}$ = -53.8880
```

The impurity occupation sector is chosen either from a cheap unrestricted Hartree–Fock
seed (`groundstate.hartree_fock_seed_occupation`; the printed dict maps orbital-symmetry
group → electron count, e.g. group 0 = $e_g$, group 1 = $t_{2g}$) or, when the seed is
rejected, from an explicit scan over occupations `{...} ~ E` where each candidate sector
is solved variationally and the lowest energy wins. `E_GS` is the winning variational
ground-state energy — compare it with `E0` further down (the final, better-converged
basis) to see how much the final expansion gained.

The report itself starts with a full-width `Ground-state report` banner; every further
part of it is introduced by a `-- <section> ----` rule (*Thermal expectation values*,
*Eigenstates*, *Correlation strength*, *Screening*, *Configurations & entanglement*,
*Density matrices*), in that fixed order. The overview under the banner collects the
scalars needed to orient in everything below:

```text
================================================================================
  Ground-state report
================================================================================
E0 = -53.888031   retained states = 3   tau = 0.0025
368 Slater determinants in the basis.
impurity spin-orbitals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Effective GS restrictions:
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] : (8, 10)
  [10, 11, ..., 103] : (82, 84)
  [40, 41, ..., 97, 104, 105] : (12, 12)
```

`basis_restrictions.get_effective_restrictions` reports, for each orbital subset
(the whole impurity, each valence-bath set, each conduction-bath set), the *observed*
minimum and maximum electron count across the converged basis. In the example: the
impurity (spin-orbitals 0–9) fluctuates between 8 and 10 electrons, one bath group is
pinned at exactly 12 (a frozen, fully occupied set), and the main valence group holds
82–84. Use this to check that the basis explored the charge fluctuations you expect —
a channel pinned at a single value contributes no screening/covalency, and a window
that hits the edge of what you configured suggests the configured restriction is
binding.

The determinant count is the size of the *variational* (CIPSI-selected) basis, not of
the full Hilbert space: only determinants that matter for the low-energy states (per-
determinant PT2 energy above `de2_min`, weight above `slaterWeightMin`) are kept. If a
memory cap bound the expansion, a truncation record is stored in `gs_info["truncation"]`
and in the JSON (`null` when the cap never bound).

## Block structure

```text
Block structure:
 0   +   +   +   +   +   +   +   +   +
 +   0   +   +   +   +   +   +   +   +
 +   +   2   +   +   +   +   +   +   +
 ...
```

`impurity spin-orbitals` (in the overview above) is the sorted list of impurity
spin-orbital indices in the computational basis — the order in which every
impurity-block matrix in this report (density matrices, spherical rotation) is sliced.

The matrix (`block_structure.print_block_structure`) shows the symmetry blocking of the
impurity one-particle space used by the Green's-function machinery. Entry $(i,j)$ is:

- an integer — orbitals $i$ and $j$ belong to a common symmetry block, and the number
  labels the *inequivalent parent* block: blocks related by symmetry (identical,
  transposed, or particle-hole conjugated) print the same number, so the set of distinct
  integers is the set of blocks that actually have to be solved;
- `+` — coupling between $i$ and $j$ is forbidden by symmetry.

In the example the ten $3d$ spin-orbitals split into ten $1\times 1$ blocks with only
two inequivalent labels, 0 ($e_g$-type) and 2 ($t_{2g}$-type): cubic symmetry with
spin degeneracy leaves two independent orbital flavours.

## Thermal expectation values

```text
Charge & occupations:
  <E>              =  -53.8880306
  <N>              =    8.2736778
  <N(Dn)>          =    4.1368389
  <N(Up)>          =    4.1368389
  <N(0,1,5,6)>     =    2.2736778
  <N(2,3,4,7,8,9)> =    6.0000000

Magnetism & multiplets:
  <Lz>             =   -0.0000000
  <Sz>             =    0.0000000
  <L.S>            =    0.0000000
  <S^2>            =    1.6624964  (S =  0.8829)
  <L^2>            =   10.3579330  (L =  2.7570)
  <J^2>            =   12.0204294  (J =  3.0029)

Spin correlations:
  <S_imp.S_bath>   =    0.0638258
```

Printed by `observables.print_thermal_expectation_values` as one aligned table in
titled sub-blocks — *Charge & occupations*, *Magnetism & multiplets* (which also
carries `<Lz+2Sz>`, `<Jz>`, `<T_z>`, the term symbol and the effective moments),
*Spin correlations*, plus the *Energy decomposition* block described below. Two
distinct kinds of quantity appear here, and telling them apart is essential for
interpretation:

**Single-particle (density-matrix) quantities** — `<N>`, `<N(Dn)>`, `<N(Up)>`, the
per-block `<N(...)>`, `<Lz>`, `<Sz>`, `<L.S>`. These are traces against the *impurity
block of the thermal single-particle density matrix* $\rho_{ij} = \langle c^\dagger_j
c_i \rangle$, rotated to the spherical-harmonics basis (`rot_to_spherical`; layout:
first $2l{+}1$ orbitals are spin-down $m_l = -l..l$, then spin-up):

- `<N>` / `<N(Dn)>` / `<N(Up)>`: total / spin-down / spin-up impurity occupation
  (`observables.get_occupations_from_rho_spherical`). Non-integer `<N>` is the norm in
  an Anderson model — it directly measures valence mixing (here $8.27$: mostly $d^8$
  with $d^9$ admixture, quantified precisely in the configuration table below).
- `<N(i,j,...)>`: occupation summed over one group of symmetry-equivalent blocks
  (labels are the block indices from the block-structure matrix; here `(0,1,5,6)` =
  the four $e_g$ spin-orbitals, `(2,3,4,7,8,9)` = the six $t_{2g}$). Use these for the
  crystal-field distribution of the charge — $t_{2g}^6 e_g^{2.27}$ in the example.
- `<Lz>`, `<Sz>` (`get_Lz/Sz_from_rho_spherical`): $\sum_{m_l} m_l\, n_{m_l}$ and
  $\tfrac12 (N_\uparrow - N_\downarrow)$. In an unpolarized calculation both must come
  out $\approx 0$ after thermal averaging over the degenerate manifold — a significantly
  non-zero value signals either intended symmetry breaking (applied field) or a manifold
  that was cut in half by `num_wanted`/the energy window.
- `<L.S>` (`get_LS_from_rho_spherical`): the expectation of the *one-body* spin-orbit
  operator, $\mathrm{Tr}(\rho\, \boldsymbol{l}\cdot\boldsymbol{s})$. It is the energy-per-unit-$\xi$
  handle on spin-orbit coupling; exactly 0 when the Hamiltonian has no SOC (as here).

Because these come from the one-particle $\rho$, they know nothing about two-particle
correlations: they cannot distinguish a spin singlet from an unpolarized triplet.

**Many-body (eigenstate) quantities** — `<S^2>`, `<L^2>`, `<J^2>`, `<S_imp.S_bath>`.
These are genuine two-body observables evaluated on the eigenstates themselves
(`observables.make_impurity_casimir_operators` builds $S^2$, $L^2$, $J^2$ for the
impurity shell in the spherical basis and rotates them to the computational basis;
`observables.apply_spin_correlation` handles $\mathbf S_\mathrm{imp}\!\cdot\!\mathbf S_\mathrm{bath}$;
both are evaluated distributed via `observables.manifold_observable_values`).

- `<S^2>` etc. are the thermal averages of $S(S{+}1)$; the parenthesised quantum number
  is $S$ recovered from the average via `casimir_to_quantum_number`. **Non-half-integer
  values are physical information, not error**: the thermal state is a weighted mixture
  of charge sectors with different spin. In the example
  $\langle S^2\rangle = 0.735\cdot 2 \,(d^8,\ S{=}1) + 0.255\cdot 0.75\,(d^9,\ S{=}\tfrac12)
  + 0.009\cdot 0\,(d^{10}) = 1.66$, i.e. $S = 0.88$ — a high-spin $d^8$ ion whose moment
  is diluted by valence fluctuations. A value locked to an exact $S(S{+}1)$ tells you the
  impurity charge is frozen; drift away from it measures mixed valence.
- `<L^2>` and `<J^2>` play the same role for the orbital moment and total moment;
  compare $S$, $L$ with the Hund's-rules values of the dominant configuration
  ($d^8$: $S{=}1$, $L{=}3$) to judge how strongly hybridization quenches them.
- `<S_imp.S_bath>` is the impurity–bath spin correlation, the Kondo-screening
  diagnostic: **negative** means antiferromagnetic impurity–bath alignment (the bath
  screens the local moment; for a fully screened $S{=}\tfrac12$ singlet it approaches
  $-3/4$), **positive** means ferromagnetic alignment, and $\approx 0$ means a free
  (unscreened) moment. It requires a consistent assignment of the bath spin-orbitals
  into (down, up) pairs; the pairing is taken from the down-then-up index convention or
  derived from the Hamiltonian's spin symmetry, and it is only *trusted* if the induced
  total-spin operator commutes with the one-body Hamiltonian
  (`spin_pairs.spin_pairs_consistent_with_h` checks $[h, S_z] = [h, S_+] = 0$).
- **Collinear spin-polarized bath** (the RSPt setup where all spin polarization lives
  in the hybridization function: spin-degenerate impurity block, spin-split bath
  energies/hoppings). Full SU(2) consistency necessarily fails there, but the weaker
  collinear check (`spin_pairs.collinear_spin_pairs_consistent_with_h`) verifies what
  *can* be verified — the global spin labelling ($[h, S_z] = 0$) and the impurity
  down↔up pairing (SU(2) of the impurity-projected one-body block) — and the report
  then prints three lines instead of one:

  - `<S_imp.S_bath> ... (transverse part depends on the bath dn/up pairing)`: the full
    correlation, using the index-convention (same fit slot) bath pairing. The spin-split
    up/down bath levels are different spatial states, so no symmetry fixes their
    pairing and the transverse ($S_\pm$) part is a modelling choice — hence the flag.
  - `<Sz_imp.Sz_bath>`: the longitudinal (Ising) part, which needs only the verified
    spin labels and is therefore **exact** — the trustworthy screening number here.
  - `cov(Sz_imp,Sz_bath)` $= \langle S^z_\mathrm{imp} S^z_\mathrm{bath}\rangle -
    \langle S^z_\mathrm{imp}\rangle\langle S^z_\mathrm{bath}\rangle$: the connected
    form. With a polarized bath both single-particle polarizations are nonzero, so the
    raw product contains a trivial static contribution; the covariance isolates the
    actual correlation (screening-cloud) part.

  The per-eigenstate table gains a matching `Szi.Szb` column in this case.
- If not even the spin labelling can be established (spin-orbit coupling, a bath
  connectivity/order the derivation cannot resolve), the lines are replaced by an
  explicit `<S_imp.S_bath> not reported: ...` message rather than a wrong number.

The Casimir lines can also be skipped (`S^2/L^2/J^2 not reported: ...`) when the
impurity is not a full spin-doubled $l$-shell; the density-matrix rows above still print.

**Magnetism and multiplet rows** (computed from the same thermal density matrix and
Casimirs; `observables.get_moments_from_rho_spherical`, `get_Tz_from_rho_spherical`,
`term_symbol`, `lande_g_and_moments`):

- `<Lz+2Sz>`: the saturation magnetic moment $m_z$ along $z$ in $\mu_B$ (the moment
  operator is $\mu_z = -\mu_B(L_z + 2S_z)$); `<Jz>` $= \langle L_z + S_z\rangle$. Both
  vanish in an unpolarized calculation — a nonzero thermal value indicates intended
  symmetry breaking (field, polarized bath) or a cut degenerate manifold.
- `<T_z>`: the intra-atomic **magnetic-dipole term**
  $T_z = \sum_i [s_z - 3\hat r_z(\hat{\mathbf r}\cdot\mathbf s)]_i$ of the XMCD spin
  sum rule (the measured "effective spin moment" is
  $\langle S_z\rangle + \tfrac{7}{2}\langle T_z\rangle$ for a d shell). Zero for a
  closed shell and for cubic site symmetry without SOC; sizeable $T_z$ warns that the
  naive spin sum rule will misread $\langle S_z\rangle$.
- `term`: the spectroscopic term symbol $^{2S+1}L_J$ recovered from the thermal
  Casimirs — e.g. `3F4` for a Hund's-rules $d^8$ ion. A `~` prefix marks values that
  are not clean (half-)integers, i.e. a mixed-valence/hybridized state where the term
  is only the nearest label.
- `g_J`, `mu_eff`, `mu_spin_only`: Landé factor
  $g_J = \tfrac32 + [\langle S^2\rangle - \langle L^2\rangle]/(2\langle J^2\rangle)$,
  effective moment $\mu_\mathrm{eff} = g_J\sqrt{\langle J^2\rangle}$ and the spin-only
  moment $2\sqrt{\langle S^2\rangle}$ (all in $\mu_B$) — the numbers to compare against
  Curie-Weiss fits of the measured susceptibility. Evaluated on the thermal Casimir
  values, so mixed valence interpolates them continuously. The `g_J`/`mu_eff` rows are
  omitted for a $J = 0$ state.

**Energy decomposition rows** (`observables.compute_energy_decomposition`; from
$\mathrm{Tr}(h_1\rho)$ over the one-body Hamiltonian blocks):

- `<H_1body>` = `<H_imp,1b>` (impurity block) + `<H_bath>` (bath block) + `<E_hyb>`
  (impurity–bath cross terms, the hybridization energy gained by covalent mixing);
- `<H_Coulomb>` $= \langle E\rangle - \langle H_\mathrm{1body}\rangle$, the interaction
  energy. For a weakly double-occupied impurity it reduces to $\approx U\!\cdot\!D$
  (with $D$ the total double occupancy below) — a quick consistency check between
  independently computed numbers.

**Static susceptibilities (Curie)** (`observables.compute_static_susceptibilities`):

```text
Static susceptibilities (Curie):
  chi_spin_zz  =   23.9411509  ((<Sz^2> - <Sz>^2)/tau)
  chi_orb_zz   =    0.0000000  ((<Lz^2> - <Lz>^2)/tau)
  chi_spin_orb =    0.0000000  ((<Sz.Lz> - <Sz><Lz>)/tau)
  chi_charge   =    4.2219966  ((<N^2> - <N>^2)/tau)
```

Fluctuation-dissipation (isothermal) susceptibilities of the impurity,
$\chi_O = \bigl(\langle O^2\rangle - \langle O\rangle^2\bigr)/\tau$, for the
longitudinal spin ($O = S_z$), orbital ($O = L_z$), their cross correlation, and the
impurity charge ($O = N$). These are the **Curie terms of the retained low-energy
manifold only**: fluctuations among the states kept by the solver, which is why they
scale as $1/\tau$ for a free moment ($\chi_{zz} = S_\mathrm{eff}(S_\mathrm{eff}+1)/3\tau$
summed over a degenerate multiplet). Van Vleck contributions from excited states above
the energy cut are *not* included — for the full frequency-resolved response (and the
screened static limit) use the dynamical susceptibility CLI
(`python -m impurityModel.ed.susceptibility`). Reading them together is the static
Hund's-metal fingerprint: a large `chi_spin_zz` next to a quenched `chi_orb_zz` and a
small `chi_charge` signals a fluctuating Hund moment with frozen orbital and charge
degrees of freedom; `chi_spin_orb` $\neq 0$ indicates spin-orbital locking. The
spin row is computed from the many-body $S_z$ of the Casimir build and must agree with
the pairs-based `chi_zz` in the correlation-diagnostics section below (two independent
implementations of the same number).

```{note}
The `<E>` row is the **absolute** thermal energy
$\langle E \rangle = \sum_n w_n E_n$ (in the example $-53.888$, which equals $E_0$ to
the digits shown because the ground manifold dominates). The relative energies appear
per state in the next table. In output produced before mid-2026 this row was labelled
`<E-E0>`, but it has always printed the absolute energy.
```

(gs-report-per-eigenstate)=
## Per-eigenstate table

```text
E0 = -53.888031
  i         E-E0         N     N(Dn)     N(Up)  N(0,1,5,6)  N(2,3,4,7,8,9)         Lz         Sz        L.S          S          L          J      Si.Sb
  0   0.00000000   8.27368   3.27368   5.00000     2.27368         6.00000   0.000000   0.863161   0.000000   0.882930   2.756982   3.002917   0.063826
  1   0.00000000   8.27368   4.65537   3.61831     2.27368         6.00000   0.000000  -0.518534   0.000000   0.882930   2.756982   3.002917   0.063826
  2   0.00000000   8.27368   4.48147   3.79221     2.27368         6.00000   0.000000  -0.344627   0.000000   0.882930   2.756982   3.002917   0.063826
```

Printed by `observables.print_expectation_values`: the same quantities as above, but
per eigenstate, plus `E0` (the absolute ground-state energy — the number to quote and
to compare across runs) and `E-E0` (excitation energies, e.g. crystal-field or
multiplet splittings).

**The degenerate-manifold caveat.** Within a (near-)degenerate manifold the eigensolver
returns an *arbitrary orthonormal basis* — any rotation of it is an equally valid set of
eigenstates. Per-state values of operators that do not commute with $H$ inside the
manifold are therefore not reproducible numbers: in the example the three degenerate
states show `Sz` $= 0.86, -0.52, -0.34$, which is one random slicing of a triplet-like
manifold. Only quantities invariant under rotations within the manifold are meaningful:
the manifold *sum/average* (here $\sum S_z \approx 0$, as printed in the thermal row),
and per-state values of operators diagonalised *within* the manifold.

The `S`, `L`, `J`, `Si.Sb` columns are of the second kind: `manifold_observable_values`
groups states that are degenerate to within `degeneracy_tol` ($10^{-6}$), builds the
small matrix $\langle m|\hat O|n\rangle$ on each manifold and diagonalises it, so these
columns are well-defined physical values (which member of the manifold carries which
value remains arbitrary when they differ). That is why `S = 0.882930` is identical and
trustworthy on all three states while `Sz` is scrambled.

Practical readings: distinct `S`/`L` between low-lying manifolds identifies competing
multiplets (e.g. a singlet–triplet gap); a jump in `N` between manifolds indicates a
different charge sector coming down in energy.

### Per-state summary

```text
Per-state summary (m_z/Jz are arbitrary within a degenerate manifold; term/S_ent are not):
  i         E-E0     Lz+2Sz         Jz      term     S_ent
  0   0.00000000  -0.96025  -0.48013     2S1/2    0.1914
```

A compact companion table (`observables.compute_state_summary`): per state the magnetic
moment `Lz+2Sz`, `Jz`, the term symbol (manifold-safe, from the per-state S/L/J values)
and the impurity–bath **entanglement entropy** `S_ent` (see the entanglement section
below; also manifold-safe). Use it to identify the character of each low-lying state at
a glance — which multiplet, how strongly entangled with the bath.

## Impurity correlation diagnostics

```text
Impurity correlation diagnostics (thermal):
    orbital(dn,up)     n_dn     n_up  <n_up*n_dn>   <m_z^2>
             (0,1)  0.97219  0.01194     0.010997   0.24053
  total double occupancy D = 0.010997
  <Sz_imp^2> = 0.240533   chi_zz = (<Sz^2> - <Sz>^2)/tau = 1.0013  (tau = 0.01)
  Inter-orbital <S_i.S_j> (...):
```

Produced by `observables.compute_correlation_diagnostics` (requires the validated spin
pairing; skipped with a message otherwise). The Mott/Hund toolbox:

- `<n_up*n_dn>` per impurity spatial orbital and the total $D$: **double occupancy**,
  the canonical correlation-strength measure — $D \to n_\uparrow n_\downarrow$
  (uncorrelated product) for weak interaction, $D \to 0$ deep in the Mott limit. The
  interaction energy above is $\sim U D$.
- `<m_z^2>` per orbital $= (\langle n_\uparrow\rangle + \langle n_\downarrow\rangle -
  2d_a)/4$: the **local moment**; $1/4$ = a fully formed spin-1/2, $0$ = empty/double
  occupied. Its size vs. the $T$-dependence of $\chi$ distinguishes local-moment from
  itinerant screening physics.
- `<Sz_imp^2>` and `chi_zz` $= (\langle S_z^2\rangle - \langle S_z\rangle^2)/\tau$: the
  static longitudinal spin susceptibility of the impurity at this $\tau$; for a free
  moment it follows the Curie law $\chi \propto 1/\tau$, saturation signals screening.
- the `<S_i.S_j>` matrix: inter-orbital spin correlations — positive off-diagonals =
  **Hund's-rule alignment** between open orbitals, negative = inter-orbital singlets.
  Diagonal entries are $\langle S_a^2\rangle = 3\langle m_z^2\rangle_a$.

## Screening channels

```text
Screening channels (thermal):
  <S_imp(0,1,5,6).S_bath> = -0.31...
  Bath levels (sorted by energy):
     bath(dn,up)    eps_dn    eps_up     n_dn     n_up   |V|_dn   |V|_up  <S_imp.S_b>
           (2,3)   -4.0000   -3.6000  0.99959  0.98795   1.0000   0.8000    -0.000309
```

Produced by `observables.compute_screening_diagnostics`. Resolves the single
`<S_imp.S_bath>` number:

- **per impurity channel** (one line per equivalent-block group, same labels as the
  `N(...)` columns; only printed when there is more than one group): which orbital
  manifold carries the screening ($e_g$ vs $t_{2g}$, …);
- **per bath level**: one row per bath (dn, up) pair with its energies (from the
  one-body $h$), thermal occupations, hybridization strengths
  $|V| = \lVert h_{\mathrm{imp},b}\rVert$ and its share of the spin correlation — the
  **screening cloud resolved over the bath spectrum**. Expect the correlation to be
  concentrated on strongly-coupled levels near the Fermi level; weight far away signals
  a mis-fitted bath. (For very large baths the correlation column is skipped and only
  the occupation table prints.)

In the collinear polarized-bath case both resolutions switch to the exact longitudinal
z-parts (labelled `Sz_imp...Sz_b`), consistent with the flagged full value.

## Ground-state statistics block

Everything between the `====` separators is produced by
`gs_statistics.compute_gs_statistics` / `print_gs_statistics` from the determinant
amplitudes of the eigenstates (computed distributed; no state vectors are gathered).
These statistics resolve *how the wavefunction is built out of Slater determinants* —
complementary to the density-matrix observables above.

Each determinant $|D\rangle$ contributes its thermally weighted amplitude square
$p_D = \sum_n w_n |\langle D|n\rangle|^2$; determinants are then bucketed by their
**configuration** — the electron-count triple

$$
(N_\mathrm{imp},\ N_\mathrm{val},\ N_\mathrm{con})
$$

in the impurity, valence-bath (occupied-side) and conduction-bath (empty-side)
orbitals.

### Thermal configuration weights

```text
Thermal ground-state configuration weights (Impurity, Valence, Conduction)
(368 Slater determinants, tau = 0.0025)
  Imp   Val   Con       Weight        %   Cumul%
    8    96     0     0.735510   73.551   73.551
    9    95     0     0.255303   25.530   99.081
   10    94     0     0.009187    0.919  100.000
```

This is the **valence histogram** of the ground state, the central covalency
diagnostic. In ligand-field language the rows are $d^8$, $d^9\underline{L}$ and
$d^{10}\underline{L}^2$ (an electron hopping from the filled valence bath onto the
impurity leaves a ligand hole $\underline{L}$). Read from it:

- the dominant configuration (is the sector you expected actually winning?);
- charge-transfer/covalent admixture: here 25.5% $d^9\underline{L}$ — sizeable
  covalency, consistent with all the fractional numbers above
  ($\langle N\rangle = 8\cdot0.736 + 9\cdot0.255 + 10\cdot0.009 = 8.27$, and the
  reduced $S = 0.88$);
- whether the basis restrictions clipped the tail: the lowest-weight row sitting at
  the edge of the effective restriction window (here `Imp` 10 at window edge (8, 10)
  with weight $10^{-2}$) is fine; a *large* weight at the window edge means the window
  is too tight.

Configurations with fractional weight below `weight_cutoff` ($10^{-3}$) are collapsed
into a `...` row showing their summed weight and count.

### Marginal occupation distributions

```text
Marginal occupation distributions (thermal):
     impurity: <N>= 8.2737  Var= 0.2172  | P(8)=0.7355  P(9)=0.2553  P(10)=0.0092
      valence: <N>=95.7263  Var= 0.2172  | P(94)=0.0092  P(95)=0.2553  P(96)=0.7355
   conduction: <N>= 0.0000  Var= 0.0000  | P(0)=1.0000
```

The same data marginalised per channel (`gs_statistics._marginal`): the probability
distribution $P(N)$, its mean and its variance. The **variance is the charge
fluctuation** $\langle N^2\rangle - \langle N\rangle^2$ of that channel:

- $\mathrm{Var} = 0$ — a frozen channel (atomic limit for the impurity; here the
  conduction baths never populate, so all charge transfer is of ligand-hole type);
- larger variance — stronger valence fluctuations / covalency; for a metallic bath
  this is the itinerancy scale of the impurity charge.

When only two channels exchange electrons their variances are identical (impurity and
valence mirror each other above) — a quick consistency check.

### Participation / entropy

```text
Participation / entropy (thermal):
  configurations: N_eff=   1.650  S=  0.6176
  determinants  : N_eff=   6.441  S=  2.3179
```

Two standard measures of how spread-out the wavefunction is over a set of
probabilities $\{p_i\}$ (`gs_statistics._participation`):

$$
N_\mathrm{eff} = \frac{1}{\sum_i p_i^2}
\qquad\text{(inverse participation ratio)},
\qquad
S = -\sum_i p_i \ln p_i
\qquad\text{(Shannon entropy, in nats)}.
$$

$N_\mathrm{eff}$ is the *effective number of entries carrying the weight*: 1 for a
single configuration/determinant, $N$ for weight spread evenly over $N$ entries. The
entropy carries the same information on a log scale, with the general bounds
$1 \le N_\mathrm{eff} \le e^{S} \le N$ ($e^{0.618} = 1.85 \ge 1.65$ in the example;
the gap between $e^S$ and $N_\mathrm{eff}$ grows when the distribution has a dominant
entry plus a long tail).

They are computed at two resolutions:

- **configurations** — over the $(N_\mathrm{imp}, N_\mathrm{val}, N_\mathrm{con})$
  buckets: measures *valence-fluctuation* character. $N_\mathrm{eff} \approx 1$ is an
  integer-valent ion; $\approx 2$ a strongly mixed-valent one (the example's 1.65 says
  "$d^8$ with substantial $d^9\underline{L}$").
- **determinants** — over the individual Slater determinants: measures the total
  *multiconfigurational* character, including the spin/orbital structure within one
  charge sector. $N_\mathrm{eff} = 6.44$ here even though the configuration count is
  1.65: the $d^8$ sector itself needs several determinants (an open $e_g^2$ shell —
  a triplet averaged over its manifold cannot be written as one determinant).

Uses: judging whether a single-reference (Hartree–Fock/DFT-like) picture of the
impurity is adequate (determinant $N_\mathrm{eff}$ close to 1) or the state is
intrinsically correlated; comparing correlation strength between parameter sets; and
sanity-checking truncation — a determinant $N_\mathrm{eff}$ comparable to the *total*
basis size means the basis has no weight hierarchy left to truncate against and the
calculation is likely under-converged.

### Impurity natural-orbital occupations

```text
Impurity natural-orbital occupations (thermal):
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  0.5684  0.5684  0.5684  0.5684
```

The eigenvalues of the impurity block of the thermal density matrix, sorted descending
— the occupations of the *natural orbitals*, the orbital basis in which $\rho$ is
diagonal. Reading them:

- occupations pinned at 1 or 0 identify inert (fully occupied / empty) orbitals —
  here the six $t_{2g}$ spin-orbitals are filled and effectively spectators;
- fractional occupations identify the *active*, correlated/hybridized orbitals — the
  four $e_g$ spin-orbitals at 0.5684 each ($4 \times 0.5684 = 2.27$ electrons, matching
  `<N(0,1,5,6)>`). Note 0.5684 is *not* close to 0 or 1: these orbitals are genuinely
  fractionally occupied, the hallmark of an open correlated shell;
- the degeneracy pattern is a symmetry check: the clean 6+4 split reflects unbroken
  cubic symmetry; split values within a manifold that should be degenerate indicate
  (intended or accidental) symmetry breaking;
- the count of fractional occupations is a good guide for choosing active spaces or
  minimal models downstream.

Two companion lines:

- `one-body entanglement entropy S_1b` $= -\sum_k [n_k\ln n_k + (1-n_k)\ln(1-n_k)]$:
  the entropy the natural occupations *would* carry if the state were free-fermion.
  Comparing it with the many-body `S_ent` below separates one-body hybridization
  entanglement from genuine interaction-induced entanglement (for the SIAM example
  above they nearly coincide — the state is almost free; a Kondo singlet has
  $S_\mathrm{ent}$ well above $S_{1b}$'s free-fermion estimate).
- `composition`: the leading components of each natural orbital in the computational
  basis — which orbitals actually mix to form the active/inert combinations (full
  vectors are stored in the JSON).

### Impurity–bath entanglement

```text
Impurity-bath entanglement (many-body impurity RDM):
  per-state S_ent = 0.1914
  thermal impurity entropy = 0.1914  (mixture entropy = 0.0000; ...)
  impurity RDM spectrum (state 0, largest): 0.9612  0.0269  0.0110  0.0009
```

Produced by `gs_statistics.compute_entanglement_entropy` from the **many-body impurity
reduced density matrix** $\rho_\mathrm{imp} = \mathrm{Tr}_\mathrm{bath}
|\psi_n\rangle\langle\psi_n|$, built distributed (determinants regrouped by bath
configuration; $\rho_\mathrm{imp}$ is block-diagonal in $N_\mathrm{imp}$, so even f
shells stay memory-bounded — a guard skips the computation with a message when the
blocks would be too large).

- `per-state S_ent` $= -\sum\lambda_i\ln\lambda_i$: for a **pure** eigenstate this is
  the genuine impurity–bath entanglement entropy — 0 for a product (atomic-limit)
  state, $\ln 2$ for a fully formed Kondo singlet of a spin-1/2, larger for
  higher-spin screening. This is the most direct single number for "how strongly is
  the impurity entangled with its environment".
- `thermal impurity entropy`: the same functional on the *thermal* RDM. For a mixed
  state it contains classical mixing as well (the printed `mixture entropy`
  $-\sum_n w_n\ln w_n$ is that part alone), so only the per-state values are pure
  entanglement measures.
- `impurity RDM spectrum`: the largest eigenvalues (the "entanglement spectrum").
  Self-check: the top eigenvalue matches the dominant configuration weight, and the
  per-$N$ block traces reproduce the impurity marginal $P(N_\mathrm{imp})$ above.

### Top Slater determinants

```text
Top Slater determinants (thermal weight):
  Imp   Val   Con       Weight        %  imp occupied | val holes | con particles
    8    96     0     0.245170   24.517  imp [2, 3, 4, 5, 6, 7, 8, 9] | val holes [] | con []
    8    96     0     0.245170   24.517  imp [0, 1, 2, 3, 4, 7, 8, 9] | val holes [] | con []
    9    95     0     0.028634    2.863  imp [0, 2, 3, 4, 5, 6, 7, 8, 9] | val holes [21] | con []
    ...
```

The individually heaviest determinants with their thermally weighted probability. The
impurity part lists the occupied spin-orbitals explicitly; the (near-filled) valence
baths are described by their *holes* and the (near-empty) conduction baths by their
occupied *particles*, which keeps the rows short — a `d^9\underline{L}` determinant
reads directly as one extra impurity index plus one valence hole. At `verbose >= 2` the
full per-channel occupation lists are printed instead. The indices are flat
computational-basis indices (the same ones appearing in `impurity spin-orbitals` and
the restriction lines). Their meaning in terms of spin/orbital character depends on the
one-body basis of the input Hamiltonian; the down-then-up pairing convention only holds
after rotation to spherical harmonics, so when in doubt derive the labelling from the
one-body Hamiltonian rather than assuming it.

This table makes the abstract statistics concrete: in the example the two leading
determinants (24.5% each) are the two $S_z=\pm 1$ members of the $e_g^2$ triplet
(occupied impurity sets `{5,6}` vs `{0,1}` on top of the filled $t_{2g}$ core), and the
next two (12.3% each) combine into its $S_z = 0$ member. Degenerate weights across
determinants related by symmetry are another quick symmetry check.

### Per-eigenstate configuration weights

```text
Per-eigenstate configuration weights (top entries):
  state 0   E-E0 =  0.000000   Boltzmann weight = 0.3333
      Imp   Val   Con       Weight        %   Cumul%
        8    96     0     0.735510   73.551   73.551
        ...
```

The configuration table resolved per eigenstate, with each state's excitation energy
and Boltzmann weight $w_n$. For a degenerate manifold the per-state tables are
identical (as here — three states at $w_n = 1/3$); differences between low-lying
states reveal competing states of different character (e.g. a $d^9$-dominated state
just above a $d^8$ ground state), which is exactly the situation where the thermal
averages must be read as mixtures.

## Impurity / bath density matrices

```text
Ground state impurity / bath density matrices:
orbital set 0:
Block 0: impurity [0, 1, 5, 6], valence [10, 11, ...], conduction []
Impurity density matrix:
    0.56842 0.00000 0.00000 0.00000
    ...
Bath density matrix (summary):
  diagonal occupations:
    [  0-  7]   1.00000   1.00000   1.00000   1.00000   1.00000   1.00000   1.00000   0.99999
    ...
  off-diagonal entries with |value| > 0.0001: 5 of 3486
    ( 10, 11) = -0.03140
    ...
```

For every orbital set and bath block, the corresponding sub-blocks of the thermal
single-particle density matrix in the computational basis. Diagonal entries are
occupations (the impurity diagonal here reproduces the natural-orbital values because
$\rho$ is already diagonal in this basis); off-diagonal entries are one-particle
coherences generated by hybridization. The impurity block is printed in full; the bath
block — mostly an identity-like diagonal — is summarized as its diagonal plus the few
significant off-diagonal coherences (indices local to the printed block), sorted by
magnitude. `verbose >= 2` prints the full bath matrix instead.

What healthy output looks like: valence-bath diagonals near 1 and conduction-bath
diagonals near 0, with deviations (and off-diagonal structure) concentrated on the few
bath orbitals coupled near the Fermi level — those are the screening channels. Bath
levels far from the Fermi level stuck at partial occupation, or large coherences to
supposedly decoupled orbitals, indicate a mislabelled bath or a symmetry problem in the
input Hamiltonian.

If anything in the reporting itself fails, the report degrades to a
`[warning] ground-state observable report incomplete ...` line — the ground state and
all returned data are unaffected; only printing was aborted.

## Worked example: reading the Ni output end to end

The snippets above come from one Ni calculation ($d^8$-like impurity, 96 bath
spin-orbitals). The full chain of reasoning a reader should be able to reproduce:

1. Configuration weights: 73.6% $d^8$, 25.5% $d^9\underline{L}$, 0.9%
   $d^{10}\underline{L}^2$ — a covalent but clearly $d^8$-dominated ground state, with
   conduction baths inert (pure ligand-hole charge transfer).
2. Consistency: $\langle N\rangle = 8(0.7355) + 9(0.2553) + 10(0.0092) = 8.274$ —
   matching `<N>`; the impurity/valence variances mirror at 0.217.
3. Spin: the sector-weighted Casimir
   $\langle S^2\rangle = 0.7355\cdot 2 + 0.2553\cdot 0.75 = 1.66$, so $S = 0.88$ — a
   Hund's-rule $d^8$ triplet ($S{=}1$) diluted by the $S{=}\tfrac12$ $d^9$ admixture,
   *not* a numerical error. Three degenerate states at Boltzmann weight $1/3$ confirm
   the (slightly quenched) triplet; their individual `Sz` values are arbitrary manifold
   slices summing to zero.
4. Orbital resolution: `<N(2,3,4,7,8,9)>` $= 6.000$ and six natural occupations at
   1.0000 — a closed $t_{2g}^6$ core; the remaining 2.27 electrons live in four $e_g$
   spin-orbitals at 0.5684 each, whose triplet structure is visible directly in the top
   determinants (two 24.5% determinants = $S_z = \pm 1$, two 12.3% = the $S_z{=}0$
   combination).
5. Correlation measures: configuration $N_\mathrm{eff} = 1.65$ (mixed valence),
   determinant $N_\mathrm{eff} = 6.4 \ll 368$ (a compact multiplet, well inside the
   variational basis — comfortably converged).
