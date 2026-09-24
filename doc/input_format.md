# Input file reference

> Generated from `impurityModel/inputformat/schema.py`; edit the `Key` declarations there, not this file. Regenerate with
> `python -m impurityModel.inputformat.schema > doc/input_format.md`.

Format version 1.0.

```bash
impurityModel init > input.toml           # a commented starter file
impurityModel run input.toml --check      # validate without solving
impurityModel run input.toml --show-resolved   # every value as the solver sees it
mpirun -n 8 impurityModel run input.toml
```

## How to read this reference

**Units.** `[units].energy` is required and has no default. It converts every key of
kind `energy` in the file, and nothing else -- a `count`, a `dimensionless` ratio and an
`inverse length` are left alone, which is why a target `occupation` (electrons) and the
NIXS `q` (reciprocal to the radial mesh) cannot be scaled by mistake. It never describes the
Hamiltonian file, which carries its own unit in its header.

**Choosing a calculation.** Write one of `[spectroscopy]`, `[selfenergy]`,
`[susceptibility]`. There is no `type = "..."` key: a tag sitting beside the tables it
names can disagree with them.

**Tagged sections.** `[hamiltonian]`, `[interaction]` and `[double_counting]` are chosen
the same way -- by which sub-table you write. Writing two is an error; leaving a stale
key from another variant is impossible, because the key lives in that variant's table.

**What you can leave out.** A self-describing `.h0` records its own bath layout, so
`n_bath` and `n_valence_bath` are deduced from it; they are required only for sources
that record no layout (a legacy `.pickle`/`.json`, a crystal-field parametrisation, a
bare matrix). A shell whose Hamiltonian is never read has no bath at all, which is the
normal case for a core shell. Every deduction is reported by `--show-resolved`.

**Angular momenta.** `l` is unrestricted here, and the solver follows: any
dipole-allowed core/valence pair is assembled on the shells declared, whether that is
an L2,3 edge (2p -> 3d), a K edge (1s -> 2p) or an M4,5 edge (3d -> 4f). What the
current solver can *run* is still checked separately, and an unsupported combination
exits saying so and naming what would have to change -- as opposed to a combination
that is wrong at any generality (a dipole transition with |l_core - l_valence| != 1 is
zero by selection rule), which is reported as invalid input.

`[hamiltonian.crystal_field]` follows the valence shell's octahedral level structure
rather than assuming a d shell: e_g / t_2g under one 10Dq for l=2, and t_1u / t_2u /
a_2u under TWO independent splittings for l=3, because the O_h invariants of an f
shell span a two-dimensional space and one number cannot place three levels. From
l=4 up there is no such parametrisation -- an irrep repeats and the point group alone
no longer fixes the basis -- so those shells need a `.h0` file.

Which keys are required follows the *model*, not just the shell. `n_bath = 0` is the
Hubbard-I approximation -- a correlated shell diagonalised with no hybridization --
and it needs only `e_imp` and its splittings; a bath block that is absent drops its
`e_val_*`/`v_val_*` or `e_con_*`/`v_con_*` rows rather than demanding values nothing
reads. Every required key must still be given. A bath block is all-or-nothing: one
partner per impurity spin-orbital, so a partial count is refused rather than halved.

**Shell roles.** `role` says which shell carries the core spin-orbit term and which
the field, the valence spin-orbit term and the double counting -- meaningful for any
pair of shells. Only XPS/XAS/RIXS build a transition operator between them, and only
those are bound by |l_core - l_valence| = 1. A PES-only, self-energy or
susceptibility run puts no constraint on the two angular momenta at all.

**Compatibility.** `[format].version` is `[major, minor]`. A newer major is refused. An
unknown key is a typo (error) when the file's minor is at or below this reader's, and a
possible future addition (warning) when it is newer. `required_features` is a hard error
either way: it is how a file says it needs a meaning this reader may not have.

## `[double_counting.amf]`

Around Mean Field. Requires the model to carry a Coulomb tensor -- there is no explicit u/j escape hatch, so it cannot run on a spectroscopy model (u4 is None there).

*No declared keys.*

## `[double_counting.fixed_gap]`

Centre dc in the charge gap (Karolak's insulator prescription): put the midpoint of the removal and addition excitations at `offset`. RECOMMENDED FOR CHARGE-TRANSFER INSULATORS, where the fixed-occupation condition breaks down. Note what is actually measured is the gap of the whole cluster, not of the impurity; the criterion reports its own exposure per edge. Runs a search.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `offset` | energy | `0.0` | Where to centre the gap; 0 is the Fermi level. |
| `guess` | energy | `0.0` | Starting double counting for the search. |
| `on_unreachable` | enum | `'abort'` | What to do when the target has no solution -- a plateau, or a target the observable steps across at a charge-sector boundary. This is the *expected* outcome of a fixed-occupation search on a charge-transfer insulator, so it is a modelling verdict, not necessarily a bug: 'keep_guess' proceeds loudly with the guess, 'abort' stops. Choices: `abort`, `keep_guess`. |
| `damping` | dimensionless | `1.0` | Mixing against the previous answer, dc = dc_prev + damping * (dc_found - dc_prev). RSPt's 'alpha' on the double-counting line, where it defaults to 0.5 because an undamped Newton step on a target that moves each CSC iteration is a limit-cycle generator. 1.0 (no damping) here, since a standalone run has no outer loop. |
| `occ_tol` | dimensionless | `0.01` | Occupation convergence tolerance. |
| `initial_step` | energy | `0.25` | First trial step of the shift search. |
| `max_shift` | energy | `20.0` | Largest |mu| the search will try before giving up. |
| `ground_state_manifold` | bool | `False` | Ask each charge sector for its degenerate ground multiplet alone instead of the whole thermal window at [temperature].tau. This criterion reads only the lowest energy of each sector, so on a model whose N +- 1 spectrum is dense inside that window the widening is bought and discarded -- SrMnO3 cubic stacks four solves per sector, ending at 160 states. Off by default because it also switches the criterion's REPORTED impurity occupation from the thermal average to the ground state's; those agree only where occupation_spread is negligible, which is not so on SrMnO3. It moves the reported mu resolution, not the root. Check the spread from a run with this off first. |
| `e_pt2_tol` | energy | `None` | Residual Epstein-Nesbet PT2 energy the CIPSI expansion is converged to: the SUMMED PT2 contribution of every determinant left out, which is what the energy error follows (it predicted the true error to 2% against exact diagonalization). THE accuracy control. Here: the charge-sector solves. Absent inherits [many_body_basis].e_pt2_tol (after the DC_E_PT2_TOL environment knob), so the double counting is measured on a space converged as far as the self-energy run's. Loosen it (e.g. 1e-5 eV) when the search is the cost. fixed_occupation takes no override: it solves on the production ground-state path and uses [many_body_basis].e_pt2_tol. |
| `de2_min` | energy | `None` | Optional per-determinant PT2 floor: candidates below it are refused whatever the residual. It bounds each refused determinant, not their sum, so it LOOSENS a solve rather than converging one -- at 1e-8 alone it left SrMnO3 5.0e-5 above its converged energy. An expansion it stops short of e_pt2_tol warns with the residual it left. Here: the charge-sector solves. Absent inherits [many_body_basis].de2_min (after the DC_DE2_MIN environment knob). |

## `[double_counting.fixed_occupation]`

Choose dc so the interacting thermal impurity occupation hits a target. Karolak's Eq. 2 -- the right criterion for METALS. Inside a gap the occupation is flat in mu, so a whole interval satisfies it and none is picked out; use fixed_gap there. Runs a search: 1-15 full collective ground-state solves.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `occupation` | dimensionless | `None` | Target impurity occupation in electrons -- NOT an energy, so [units].energy does not touch it. Absent means the DFT reference filling of the raw h0. |
| `guess` | energy | `0.0` | Starting double counting for the search. |
| `on_unreachable` | enum | `'abort'` | What to do when the target has no solution -- a plateau, or a target the observable steps across at a charge-sector boundary. This is the *expected* outcome of a fixed-occupation search on a charge-transfer insulator, so it is a modelling verdict, not necessarily a bug: 'keep_guess' proceeds loudly with the guess, 'abort' stops. Choices: `abort`, `keep_guess`. |
| `damping` | dimensionless | `1.0` | Mixing against the previous answer, dc = dc_prev + damping * (dc_found - dc_prev). RSPt's 'alpha' on the double-counting line, where it defaults to 0.5 because an undamped Newton step on a target that moves each CSC iteration is a limit-cycle generator. 1.0 (no damping) here, since a standalone run has no outer loop. |
| `occ_tol` | dimensionless | `0.01` | Occupation convergence tolerance. |
| `initial_step` | energy | `0.25` | First trial step of the shift search. |
| `max_shift` | energy | `20.0` | Largest |mu| the search will try before giving up. |

## `[double_counting.fixed_peak]`

Choose dc so a peak in the impurity spectral function lands at a given energy. Positive places an electron-addition peak, negative a removal peak. Runs a search.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `peak_position` | energy | **required** | Where to put the peak, relative to E_F. |
| `guess` | energy | `0.0` | Starting double counting for the search. |
| `on_unreachable` | enum | `'abort'` | What to do when the target has no solution -- a plateau, or a target the observable steps across at a charge-sector boundary. This is the *expected* outcome of a fixed-occupation search on a charge-transfer insulator, so it is a modelling verdict, not necessarily a bug: 'keep_guess' proceeds loudly with the guess, 'abort' stops. Choices: `abort`, `keep_guess`. |
| `damping` | dimensionless | `1.0` | Mixing against the previous answer, dc = dc_prev + damping * (dc_found - dc_prev). RSPt's 'alpha' on the double-counting line, where it defaults to 0.5 because an undamped Newton step on a target that moves each CSC iteration is a limit-cycle generator. 1.0 (no damping) here, since a standalone run has no outer loop. |
| `occ_tol` | dimensionless | `0.01` | Occupation convergence tolerance. |
| `initial_step` | energy | `0.25` | First trial step of the shift search. |
| `max_shift` | energy | `20.0` | Largest |mu| the search will try before giving up. |
| `ground_state_manifold` | bool | `False` | Ask each charge sector for its degenerate ground multiplet alone instead of the whole thermal window at [temperature].tau. This criterion reads only the lowest energy of each sector, so on a model whose N +- 1 spectrum is dense inside that window the widening is bought and discarded -- SrMnO3 cubic stacks four solves per sector, ending at 160 states. Off by default because it also switches the criterion's REPORTED impurity occupation from the thermal average to the ground state's; those agree only where occupation_spread is negligible, which is not so on SrMnO3. It moves the reported mu resolution, not the root. Check the spread from a run with this off first. |
| `e_pt2_tol` | energy | `None` | Residual Epstein-Nesbet PT2 energy the CIPSI expansion is converged to: the SUMMED PT2 contribution of every determinant left out, which is what the energy error follows (it predicted the true error to 2% against exact diagonalization). THE accuracy control. Here: the charge-sector solves. Absent inherits [many_body_basis].e_pt2_tol (after the DC_E_PT2_TOL environment knob), so the double counting is measured on a space converged as far as the self-energy run's. Loosen it (e.g. 1e-5 eV) when the search is the cost. fixed_occupation takes no override: it solves on the production ground-state path and uses [many_body_basis].e_pt2_tol. |
| `de2_min` | energy | `None` | Optional per-determinant PT2 floor: candidates below it are refused whatever the residual. It bounds each refused determinant, not their sum, so it LOOSENS a solve rather than converging one -- at 1e-8 alone it left SrMnO3 5.0e-5 above its converged energy. An expansion it stops short of e_pt2_tol warns with the residual it left. Here: the charge-sector solves. Absent inherits [many_body_basis].de2_min (after the DC_DE2_MIN environment knob). |

## `[double_counting.fll]`

Fully Localized Limit, dc = [U(N - 1/2) - (J/2)(N - 1)] I, at the DFT reference occupation. Needs U and J: derived from the Coulomb tensor when the model has one, otherwise supply them here.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `u` | energy | `None` | Average Coulomb repulsion; derived from u4 when absent. |
| `j` | energy | `None` | Average exchange; derived from u4 when absent. |

## `[double_counting.mlft]`

RSPt's charge-transfer correction `c`. SPECTROSCOPY ONLY, and not a double-counting matrix: it enters H with a `+` sign folded into h0 by ImpurityModel.from_shells and takes a different value per shell, whereas every scheme below produces a matrix that is SUBTRACTED. Sharing one `value` key between the two would be a sign error waiting to happen, which is why this has its own tag.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `c` | energy | `1.5` | The charge-transfer correction. |

## `[double_counting.nominal]`

FLL evaluated at the NOMINAL integer occupation rather than the DFT reference. Needs no reference filling, so it cannot saturate on a coarse bath fit -- the natural first guess, and a reference to check a converged fixed-occupation answer against.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `u` | energy | `None` | Average Coulomb repulsion; derived from u4 when absent. |
| `j` | energy | `None` | Average exchange; derived from u4 when absent. |

## `[double_counting.none]`

No double counting.

*No declared keys.*

## `[double_counting.sigma_inf]`

The static (high-frequency) limit of the self-energy. Requires a Coulomb tensor.

*No declared keys.*

## `[environment]`

Runtime tuning knobs, by their registry name in impurityModel.ed.config. Free-form: every key is validated against that registry, so an unknown name gets an exact closest-match suggestion rather than a guess. Reachable from the RSPt interface too, which is why it is a table of its own rather than CLI flags.

*No declared keys.*

## `[format]`

Format version and forward-compatibility declarations.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `version` | version | `[1, 0]` | [major, minor]. A major above the reader's is refused outright. The minor decides how an unknown key is treated: at or below ours it can only be a typo (error), above ours it may be a future key (warn and ignore). |
| `required_features` | string list | `[]` | Semantics a reader must understand to interpret this file correctly. An entry this reader does not recognise is a hard error -- the same contract as the .h0 header (doc/h0_file_format.md), and the reason unknown keys can safely be lenient while this is strict. |

## `[hamiltonian.archive]`

Reconstruct the model from an impurityModel_data.h5 archive written by the RSPt interface. The archive supplies the model, both frequency meshes and the recorded basis/solver options, so tables it covers must not also appear in this file.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `path` | path | **required** | Archive file. |
| `cluster` | string | `None` | Cluster label; default is the first group. |
| `iteration` | count | `None` | DMFT iteration; default is the last. |

## `[hamiltonian.blocks]`

Build from the impurity / hybridization / bath blocks, H = [[H_imp, V^dag], [V, H_bath]]. The impurity block is the valence shell's 2(2l+1) spin-orbitals in the (l, s, m) layout, [interaction.slater] F_vv supplies the interaction, and a non-zero shell soc or zeeman_splitting is refused (the matrix does not state its basis): fold it into the matrix.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `h_imp` | matrix | **required** | Effective impurity block (n_imp, n_imp). |
| `v` | matrix | **required** | Impurity-bath hopping (n_bath, n_imp). |
| `h_bath` | matrix | **required** | Bath block (n_bath, n_bath). |

## `[hamiltonian.crystal_field]`

Build the Hamiltonian from an octahedral crystal-field parametrisation. Which keys are required depends on the valence shell's l, because its O_h level structure does: a d shell has two levels (e_g, t_2g) split by one 10Dq, an f shell has three (t_1u, t_2u, a_2u) and needs TWO independent splittings -- one number cannot place three levels. The required set is checked when the shell is known, and EVERY key in it must be given: the underlying reader fills an absent d-shell key from a hard-coded Ni-in-NiO value, so the shipped CoO/FeO/MnO files (which set six) silently ran with Ni's conduction bath.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `e_imp` | energy | `None` | Average valence-shell on-site energy. |
| `e_deltaO_imp` | energy | `None` | Rank-4 octahedral splitting of the valence shell -- 10Dq for a d shell. Each splitting key is the full spread (highest level minus lowest) that invariant alone produces. Required for l >= 2; an l = 0 or 1 shell is a single O_h level and has no splitting at all. |
| `e_delta6_imp` | energy | `None` | Rank-6 octahedral splitting, for l = 3 only. Independent of e_deltaO_imp: the two invariants place the f shell's three levels between them (t_1u : t_2u : a_2u = 3 : -1 : -6 for rank 4, 5 : -9 : 12 for rank 6). SIGN: both parameters carry the sign a point-charge octahedron produces, so a real octahedral field has both POSITIVE. The Stevens form B4(O_4^0 + 5 O_4^4) + B6(O_6^0 - 21 O_6^4) does not -- an octahedron gives B4 > 0 but B6 < 0 -- so e_deltaO_imp follows B4's sign and e_delta6_imp is the OPPOSITE of B6's. Importing B6 from a paper means flipping it. To set both from two level splittings instead, with d1 = E(t_1u) - E(a_2u) and d2 = E(t_2u) - E(a_2u): e_deltaO_imp = 9*(3*d1 - d2)/22, e_delta6_imp = 3*(e_deltaO_imp - d1). |
| `e_val_a1g` | energy | `None` | Valence bath level coupled to the a1g orbitals. Give this for a valence shell whose octahedral levels include a1g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_val_t1u` | energy | `None` | Valence bath level coupled to the t1u orbitals. Give this for a valence shell whose octahedral levels include t1u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_val_eg` | energy | `None` | Valence bath level coupled to the eg orbitals. Give this for a valence shell whose octahedral levels include eg (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_val_t2g` | energy | `None` | Valence bath level coupled to the t2g orbitals. Give this for a valence shell whose octahedral levels include t2g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_val_t2u` | energy | `None` | Valence bath level coupled to the t2u orbitals. Give this for a valence shell whose octahedral levels include t2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_val_a2u` | energy | `None` | Valence bath level coupled to the a2u orbitals. Give this for a valence shell whose octahedral levels include a2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_a1g` | energy | `None` | Conduction bath level coupled to the a1g orbitals. Give this for a valence shell whose octahedral levels include a1g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_t1u` | energy | `None` | Conduction bath level coupled to the t1u orbitals. Give this for a valence shell whose octahedral levels include t1u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_eg` | energy | `None` | Conduction bath level coupled to the eg orbitals. Give this for a valence shell whose octahedral levels include eg (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_t2g` | energy | `None` | Conduction bath level coupled to the t2g orbitals. Give this for a valence shell whose octahedral levels include t2g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_t2u` | energy | `None` | Conduction bath level coupled to the t2u orbitals. Give this for a valence shell whose octahedral levels include t2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `e_con_a2u` | energy | `None` | Conduction bath level coupled to the a2u orbitals. Give this for a valence shell whose octahedral levels include a2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_a1g` | energy | `None` | Valence hybridization with the a1g orbitals. Give this for a valence shell whose octahedral levels include a1g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_t1u` | energy | `None` | Valence hybridization with the t1u orbitals. Give this for a valence shell whose octahedral levels include t1u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_eg` | energy | `None` | Valence hybridization with the eg orbitals. Give this for a valence shell whose octahedral levels include eg (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_t2g` | energy | `None` | Valence hybridization with the t2g orbitals. Give this for a valence shell whose octahedral levels include t2g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_t2u` | energy | `None` | Valence hybridization with the t2u orbitals. Give this for a valence shell whose octahedral levels include t2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_val_a2u` | energy | `None` | Valence hybridization with the a2u orbitals. Give this for a valence shell whose octahedral levels include a2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_a1g` | energy | `None` | Conduction hybridization with the a1g orbitals. Give this for a valence shell whose octahedral levels include a1g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_t1u` | energy | `None` | Conduction hybridization with the t1u orbitals. Give this for a valence shell whose octahedral levels include t1u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_eg` | energy | `None` | Conduction hybridization with the eg orbitals. Give this for a valence shell whose octahedral levels include eg (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_t2g` | energy | `None` | Conduction hybridization with the t2g orbitals. Give this for a valence shell whose octahedral levels include t2g (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_t2u` | energy | `None` | Conduction hybridization with the t2u orbitals. Give this for a valence shell whose octahedral levels include t2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `v_con_a2u` | energy | `None` | Conduction hybridization with the a2u orbitals. Give this for a valence shell whose octahedral levels include a2u (d: e_g, t_2g; f: t_1u, t_2u, a_2u; s: a_1g; p: t_1u). |
| `bath_state_basis` | enum | `'spherical'` | Basis the bath states are expressed in. Reachable from no CLI today. Choices: `spherical`, `cubic`. |

## `[hamiltonian.file]`

Read the one-particle Hamiltonian from a file: a self-describing flat `.h0`, or a legacy labelled `.pickle`/`.json`/`.dat`. Which one is decided by the file's own content, not its extension (see ed.model.load_model).

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `path` | path | **required** | The Hamiltonian file, relative to this input file. |
| `unit` | enum | `None` | ERROR on a legacy format: nothing in the reader scales a pickle/.dat/.json amplitude, every shipped legacy file is already eV-scale, and anyone holding a Rydberg Hamiltonian is on .h0, which records its own unit. Convert to .h0 instead. On a .h0 this may only restate the header's unit; disagreeing is an error, never a silent override. Choices: `eV`, `Ry`, `Ha`. |
| `n_impurity_orbitals` | count | `None` | Impurity block size, for the legacy bare-integer format only -- it records no orbital layout. Validated against the file's sparsity pattern. |
| `contains_soc` | bool | `None` | Cross-check against a .h0 header, never an override. The header treats an absent value as *unknown*, not false, and requesting a non-zero shell `soc` against an unknown or true value is a hard error -- this exact SOC double-counting has shipped once already. |
| `energy_reference` | enum | `None` | Cross-check against the header. 'absolute' is refused for any double-counting scheme, sector walk or Fermi-centred mesh: the bath valence/conduction split is taken from sign(h[o,o]) and the DFT reference filling from mu_chem = 0, so an offset zero silently re-partitions the bath into a different model. Choices: `fermi`, `absolute`. |
| `spin` | enum | `'explicit'` | How the file's orbitals relate to spin. 'explicit': every orbital is a spin-orbital, as written. 'degenerate': the file holds SPATIAL orbitals only (a spinless model Hamiltonian), and each is copied to both spins, spin down first -- impurity block [down, up], then the bath's down copies, then its up copies. .h0 files only. Never inferred from an odd impurity block: a file missing its second spin would otherwise be silently repaired into a different model. Choices: `explicit`, `degenerate`. |

## `[hamiltonian.matrix]`

Build from the full one-particle solver matrix, impurity block first. The impurity block is the valence shell's 2(2l+1) spin-orbitals in the (l, s, m) layout, [interaction.slater] F_vv supplies the interaction, and a non-zero shell soc or zeeman_splitting is refused (the matrix does not state its basis): fold it into the matrix.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `h` | matrix | **required** | Full (n, n) one-particle Hamiltonian. |
| `n_impurity_orbitals` | count | **required** | Leading impurity block dimension. |

## `[interaction.core]`

Core-shell Slater-Condon integrals for a spectroscopy run whose VALENCE interaction is one of the model forms (kanamori, density_density, terms, u4_file). With [interaction.slater] give these there instead; declaring both is an error.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `F_cc` | energy list | `None` | Core-core F^k. Length 2*l_c + 1. |
| `F_cv` | energy list | `None` | Core-valence direct F^k. Length 2*min(l_v, l_c) + 1. |
| `G_cv` | energy list | `None` | Core-valence exchange G^k. Length l_v + l_c + 1. |

## `[interaction.density_density]`

Density-density interaction, H = sum_(a,b) U_opp[a,b] n_a,up n_b,dn + sum_(a<b,s) U_same[a,b] n_a,s n_b,s. Both matrices symmetric; each unordered same-spin pair is counted once.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `U_opposite_spin` | energy matrix | **required** | n x n opposite-spin repulsion; its diagonal is the intra-orbital Hubbard U. |
| `U_same_spin` | energy matrix | `None` | n x n same-spin repulsion, zero diagonal (Pauli). Zero when absent. |
| `orbital_basis` | enum | `None` | Which orbitals the indices refer to, on an l >= 1 shell (required there, meaningless on a model shell or l = 0). The interaction is defined among REAL orbitals and is not invariant under a complex rotation, so this cannot be defaulted. 'real_cubic': the cubic harmonics, in the O_h level order of atomic_physics.get_spherical_2_cubic_matrix (d: e_g, e_g, t_2g, t_2g, t_2g); the tensor is rotated to the shell's spherical (l, s, m) basis. 'as_hamiltonian': the Hamiltonian file's own impurity orbitals, in its order. Choices: `real_cubic`, `as_hamiltonian`. |

## `[interaction.kanamori]`

Hubbard-Kanamori interaction on the valence shell's n orbitals: H = U sum_a n_a,up n_a,dn + U' sum_(a!=b) n_a,up n_b,dn + (U'-J) sum_(a<b,s) n_a,s n_b,s - J sum_(a!=b) c+_a,up c_a,dn c+_b,dn c_b,up + J_pair sum_(a!=b) c+_a,up c+_a,dn c_b,dn c_b,up. One orbital is the single-band Hubbard U n_up n_dn. Each parameter is independently settable, so the rotationally invariant point is a default rather than a constraint.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `U` | energy | **required** | Intra-orbital repulsion. |
| `J` | energy | `0.0` | Hund's exchange: the spin flip, and the same-spin reduction U' - J. |
| `U_prime` | energy | `None` | Inter-orbital repulsion. Defaults to U - 2J. |
| `J_pair` | energy | `None` | Pair hopping. Defaults to J. |
| `orbital_basis` | enum | `None` | Which orbitals the indices refer to, on an l >= 1 shell (required there, meaningless on a model shell or l = 0). The interaction is defined among REAL orbitals and is not invariant under a complex rotation, so this cannot be defaulted. 'real_cubic': the cubic harmonics, in the O_h level order of atomic_physics.get_spherical_2_cubic_matrix (d: e_g, e_g, t_2g, t_2g, t_2g); the tensor is rotated to the shell's spherical (l, s, m) basis. 'as_hamiltonian': the Hamiltonian file's own impurity orbitals, in its order. Choices: `real_cubic`, `as_hamiltonian`. |

## `[interaction.none]`

No interaction: a non-interacting reference calculation.

*No declared keys.*

## `[interaction.slater]`

Slater-Condon parameters. Array lengths are DERIVED from the shells' angular momenta (2*l_v+1, 2*l_c+1, 2*l_c+1, 2*l_c+2) and checked, rather than restated as l_core / l_valence keys -- one source of truth per angular momentum.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `F_vv` | energy list | **required** | Valence-valence F^k (was Fdd). Length 2*l_v + 1. |
| `F_cc` | energy list | `None` | Core-core F^k (was Fpp). Length 2*l_c + 1. |
| `F_cv` | energy list | `None` | Core-valence direct F^k (was Fpd), indexed by k. Length 2*min(l_v, l_c) + 1, which is the familiar 2*l_c + 1 whenever the core shell is the lower one. |
| `G_cv` | energy list | `None` | Core-valence exchange G^k (was Gpd), indexed by k. Length l_v + l_c + 1 -- the same as the familiar 2*l_c + 2 at every dipole-allowed edge, and longer only for a shell pair more than one apart, which the roles allow but no transition operator does. |

## `[interaction.terms]`

Explicit interaction matrix elements <pq|V|rs> (physicists' notation: electron 1 goes p -> r, electron 2 goes q -> s), H = 1/2 sum <pq|V|rs> c+_p c+_q c_s c_r. Each entry is [p, q, r, s, value] or [p, q, r, s, re, im]. The fully general form, for teaching and for interactions no parametrisation covers.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `spatial` | energy terms | `None` | Elements between SPATIAL orbitals 0..n-1, expanded over spin so that the interaction is spin-rotation invariant: <pq|V|rs> applies to every spin pair, spin carried along each electron line. U n_up n_dn on orbital 0 is [0, 0, 0, 0, U]. |
| `spin_orbital` | energy terms | `None` | Elements between SPIN-orbitals 0..2n-1 (spin-orbital s*n + p, spin down first), taken literally: for spin-dependent interactions. Added to the expanded `spatial` list. |
| `complete_symmetries` | bool | `True` | Fill in each element's images <qp|V|sr> = <pq|V|rs> and <rs|V|pq> = <pq|V|rs>*, so each distinct element is written once. Writing an image as well is harmless (it is filled, never added); writing it with a DIFFERENT value is an error. false takes the lists literally, which is rarely what you want: a missing Hermitian partner is refused. |
| `orbital_basis` | enum | `None` | Which orbitals the indices refer to, on an l >= 1 shell (required there, meaningless on a model shell or l = 0). The interaction is defined among REAL orbitals and is not invariant under a complex rotation, so this cannot be defaulted. 'real_cubic': the cubic harmonics, in the O_h level order of atomic_physics.get_spherical_2_cubic_matrix (d: e_g, e_g, t_2g, t_2g, t_2g); the tensor is rotated to the shell's spherical (l, s, m) basis. 'as_hamiltonian': the Hamiltonian file's own impurity orbitals, in its order. Choices: `real_cubic`, `as_hamiltonian`. |

## `[interaction.u4_file]`

Read the four-index Coulomb tensor from a .npy file, in [units].energy, in RSPt convention: u4[i,j,k,l] = <ij|V|kl>, H = 1/2 sum u4[i,j,k,l] c+_i c+_j c_l c_k. For a tensor too large to write out; small ones read better as [interaction.terms].

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `path` | path | **required** | The .npy file. |
| `index_space` | enum | `'spin_orbital'` | 'spin_orbital': the tensor spans the 2n impurity spin-orbitals (spin-major, down first), taken as written. 'spatial': it spans the n spatial orbitals and is expanded over spin, like [interaction.terms].spatial. Choices: `spin_orbital`, `spatial`. |
| `orbital_basis` | enum | `None` | Which orbitals the indices refer to, on an l >= 1 shell (required there, meaningless on a model shell or l = 0). The interaction is defined among REAL orbitals and is not invariant under a complex rotation, so this cannot be defaulted. 'real_cubic': the cubic harmonics, in the O_h level order of atomic_physics.get_spherical_2_cubic_matrix (d: e_g, e_g, t_2g, t_2g, t_2g); the tensor is rotated to the shell's spherical (l, s, m) basis. 'as_hamiltonian': the Hamiltonian file's own impurity orbitals, in its order. Choices: `real_cubic`, `as_hamiltonian`. |

## `[many_body_basis]`

How the many-body determinant basis is built. Named for the determinant basis specifically: 'basis' alone means both the single-particle orbital basis (a .h0 header declares one) and this, and both appear in one input file.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `truncation_threshold` | auto/count | `'auto'` | Cap on determinants per basis. 'auto' derives it from available per-rank memory at the (collective) call site; 'none' disables capping. The two are NOT interchangeable even though the underlying code currently collapses both to infinity in one place. Choices: `auto`, `none`. |
| `excitation_budget` | auto/count | `'auto'` | Maximum total bath excitations per determinant. 'auto' takes the solver's measured-lossless default; 'none' disables it. Prefer omitting to writing the number: the default is documented as the tightest MEASURED value and is expected to be re-measured, so a copy here would freeze a stale one. Choices: `auto`, `none`. |
| `chain_restrict` | bool | `True` | Apply chain occupation restrictions. |
| `occ_cutoff` | dimensionless | `None` | Occupation cutoff deciding filled/partial/empty bath classification, i.e. the variational space -- not cosmetic. Per-calculation default. |
| `slater_weight_min` | dimensionless | `None` | Minimum determinant weight retained. |
| `e_pt2_tol` | energy | `None` | Residual Epstein-Nesbet PT2 energy the CIPSI expansion is converged to: the SUMMED PT2 contribution of every determinant left out, which is what the energy error follows (it predicted the true error to 2% against exact diagonalization). THE accuracy control. Here: the production ground state, and the default for the double-counting search. Absent is the solver's default, 1e-8 eV. Must be positive. |
| `de2_min` | energy | `None` | Optional per-determinant PT2 floor: candidates below it are refused whatever the residual. It bounds each refused determinant, not their sum, so it LOOSENS a solve rather than converging one -- at 1e-8 alone it left SrMnO3 5.0e-5 above its converged energy. An expansion it stops short of e_pt2_tol warns with the residual it left. Here: the production ground state, and the default for the double-counting search. Absent is no floor. |
| `dN` | count | `None` | Impurity occupation window (+-dN) for the excited bases. Note the sentinel means different things per driver: the spectroscopy path substitutes 2, the Green's-function path treats absent as NO window at all. |
| `mixed_valence` | dimensionless | `None` | Mixed-valence scalar, forwarded per group. |

## `[rotation_to_spherical]`

Rotation from the impurity basis to spherical harmonics. Used for L/S/J OBSERVABLE REPORTING ONLY -- it does not rotate the Hamiltonian into a spherical representation, and the solver composes its own rotation independently. Stored per shell, so a per-shell override is a sub-table.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `from_h0` | bool | `True` | Take the rotation from the .h0 header, falling back to the identity. Set false to require an explicit per-shell matrix. |

## `[run]`

Where output goes and how much of it there is.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `outdir` | output path | `'.'` | Directory for the output archive, relative to where you run from -- NOT to this file, unlike every input path. Where results go is a property of the invocation; running one input file from two directories should write two sets of results, not fight over one. |
| `verbosity` | count | `0` | 0-3; the CLI's -v/-vv/-vvv overrides this. |

## `[selfenergy]`

Impurity self-energy Sigma(w) / Sigma(i nu) and the impurity Green's function.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `cluster` | string | `'cluster'` | Cluster label used in the output filenames. |
| `output` | output path | `None` | Output archive; default selfenergy-<cluster>.h5. |

## `[selfenergy.matsubara]`

FERMIONIC Matsubara output: i*nu_n with nu_n = (2n+1)*pi*tau.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | Compute it. An explicit switch, not a zero count. |
| `n_points` | count | `0` | Number of fermionic Matsubara frequencies. |

## `[selfenergy.real_axis]`

Real-frequency output.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. An explicit switch, not an empty mesh. |
| `mesh` | mesh | `{'min': -10.0, 'max': 10.0, 'n': 2001}` | Real frequencies, relative to E_F. |
| `broadening` | energy | `0.1` | Distance above the real axis. |

## `[[shell]]`

One correlated or core shell. An array of tables, so a shell's angular momentum is tied to ITS OWN bath count and occupation -- unlike the CLI's four order-coupled lists (--ls / --nBaths / --nValBaths / --n0imps), where only list position relates them and only equal lengths are checked.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `l` | count | `None` | Angular momentum. UNRESTRICTED by this schema, and the solver now follows: any dipole-allowed (core l, valence l) pair is assembled on the shells declared here. What the solver can actually do is still checked separately -- see inputformat.capabilities. Give exactly one of `l` and `n_orbitals`. |
| `n_orbitals` | count | `None` | Number of spatial orbitals of a MODEL shell -- one with no angular momentum, such as a single-orbital Anderson model or a two-orbital Hubbard model. The shell has 2*n_orbitals spin-orbitals, spin down first. Instead of `l`, never with it: an l shell has 2l+1 orbitals and a spherical structure, a model shell only has its count. Valence shells only, and not for spectroscopy, whose transition operators need l; nor with [interaction.slater], whose integrals are defined on an l shell. |
| `role` | enum | **required** | REQUIRED and never inferred from `l`. The inference 'l=1 means core, l=2 means valence' is precisely the hardcoding this format has to outlive. Choices: `core`, `valence`. |
| `n_bath` | count | *deduced* | Total bath states for this shell. Deduced from the .h0 header (n_orb minus the impurity block) for the shell the file describes; 0 for every other shell, since a shell with no Hamiltonian has no fitted bath -- the normal case for a core shell. Required for any non-.h0 source, none of which records a bath layout. |
| `n_valence_bath` | count | *deduced* | Bath states that start occupied. Must not exceed n_bath. Deduced from the .h0 header's valence_bath/conduction_bath lists when present; otherwise from the bath on-site energies, h[o,o] < 0 being valence -- the same rule solver_basis.classify_bath_occupation already applies. 0 for a shell the file does not describe. |
| `nominal_occupation` | count | **required** | Nominal electron count on this shell. |
| `soc` | energy | `0.0` | Spin-orbit coupling to add. Only added when the Hamiltonian does not already contain it; a non-zero value against a .h0 whose header says contains_soc is true, or does not say at all, is a hard error. |
| `zeeman_splitting` | energy vector | `None` | Zeeman ENERGY (hx, hy, hz) -- a spin-only splitting with no Bohr magneton, no g-factor and no orbital term, so it is not 'a magnetic field'. Omitting it means NO FIELD, on every Hamiltonian format. The underlying readers each have their own default (the labelled formats apply a (0, 0, 1e-4) symmetry-breaking nudge, the flat one applies nothing), which would make an omitted key mean different physics depending on the input file; this format does not inherit that. Ask for a field if you want one. |

## `[solver]`

Green's-function kernel and eigensolver settings.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `gf_method` | enum | `'lanczos'` | Green's-function kernel. Choices: `lanczos`, `bicgstab`, `sliced`, `cipsi`. |
| `reort` | auto/enum | `'auto'` | Block-Lanczos reorthogonalization. 'auto' is the solver's own default, which is NOT one mode: it means NONE on the Green's-function path and PARTIAL on the eigensolver path. Writing a mode also moves the derived determinant budget, since retention switches the memory model to its worst case. Choices: `auto`, `none`, `partial`, `periodic`, `selective`, `full`. |
| `dense_cutoff` | count | `500` | Use a dense eigensolver below this matrix size. |
| `sparse_green` | bool | `True` | Use the sparse block-Lanczos Green's-function path. |
| `auto_block_structure` | bool | `True` | Derive the block structure and symmetry-adapted solver basis from the hybridization-dressed impurity matrix instead of the fall-back one block per shell. A solver-basis decision (it replaces the Hamiltonian operator the solve runs on), which is why it lives here and not under a spectroscopy table. |

## `[spectroscopy]`

PES / XPS / XAS / RIXS / NIXS. The meshes and the core-hole broadening live HERE, not under a technique, because the code genuinely shares them: one `delta` is both the PES/XPS/XAS lineshape and RIXS's intermediate-state broadening, and NIXS is evaluated on RIXS's energy-loss mesh. Filing either under one technique would mean switching that technique off changed another one.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `w` | mesh | `{'min': -25.0, 'max': 25.0, 'n': 3001}` | PES / XPS / XAS evaluation mesh, relative to E_F. |
| `w_loss` | mesh | `{'min': -2.0, 'max': 12.0, 'n': 4000}` | Energy-loss mesh, shared by RIXS and NIXS. |
| `core_hole_broadening` | energy | `0.2` | HWHM above the real axis. Sets the PES/XPS/XAS lineshape AND the RIXS INTERMEDIATE-state resolvent broadening -- one number, two roles, which is why it is not named per technique. |
| `cluster` | string | `'cluster'` | Label used in the output. |
| `output` | output path | `'spectra.h5'` | Output archive, relative to [run].outdir. |

## `[spectroscopy.nixs]`

Non-resonant inelastic x-ray scattering, on the shared w_loss mesh.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | Compute it. Previously implied by supplying a radial file; now explicit, and the radial file is required when this is on. |
| `radial_file` | path | `None` | Two-column radial wavefunction of the correlated orbitals. Its length unit is what makes `q` meaningful -- they are reciprocal. |
| `broadening` | energy | `0.1` | HWHM for NIXS. |
| `q` | vector list | `None` | Momentum transfers, reciprocal to the radial mesh's length unit -- an inverse length, so [units].energy does not touch it. Any direction is fine, the pole included; |q| = 0 is refused, having no scattering direction. |
| `l_final` | count | `None` | Angular momentum of the final orbitals (was liNIXS). Omitted means the valence shell's l -- NIXS is a valence probe, so a fixed default of 2 would silently give a non-d model the d shell's angular momentum. |
| `l_initial` | count | `None` | Angular momentum of the initial orbitals (was ljNIXS). Omitted means the valence shell's l. |

## `[spectroscopy.pes]`

Valence photoemission and inverse photoemission.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. Today this is unconditional and cannot be switched off. |

## `[spectroscopy.rixs]`

Resonant inelastic x-ray scattering, on the shared w_loss mesh.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | Compute it. THE ONLY SWITCH: a non-positive broadening and an empty incoming mesh used to disable RIXS as side effects, which meant two independent switches with no stated precedence and a broadening doubling as a feature flag. Both are now validation errors instead. |
| `w_in` | mesh | `{'min': -10.0, 'max': 20.0, 'n': 50}` | Incoming photon energies. |
| `final_state_broadening` | energy | `0.05` | HWHM of the FINAL state. The intermediate-state half of the lineshape is the shared core_hole_broadening. |

## `[spectroscopy.xas]`

X-ray absorption. Uses the shared core_hole_broadening.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. |

## `[spectroscopy.xps]`

Core-level photoemission.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. Today this is unconditional and cannot be switched off. |

## `[susceptibility]`

Dynamical impurity susceptibilities chi(w) / chi(i nu).

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `cluster` | string | `'cluster'` | Cluster label used in the output. |
| `output` | output path | `'chi.h5'` | Output archive, relative to [run].outdir. |

## `[susceptibility.matsubara]`

BOSONIC Matsubara output. Distinct from the self-energy's in both statistics and convention (this mesh is real-valued and includes nu = 0, which carries the Van Vleck term), which is why the two are separate tables rather than one shared key.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. |
| `n_points` | count | `64` | Number of bosonic Matsubara frequencies. |

## `[susceptibility.real_axis]`

Real-frequency output.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `True` | Compute it. |
| `mesh` | mesh | `{'min': -5.0, 'max': 5.0, 'n': 501}` | Real frequencies. |
| `broadening` | energy | `0.01` | Distance above the real axis. |

## `[temperature]`

The thermal occupation. Give exactly one of these: they are governed by different units, and one table carrying two unit governances is how a `tau = 0.002` under [units].energy = 'Ry' becomes a silent 13.6x temperature error.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `kelvin` | temperature | `None` | Temperature; Kelvin unless [units].temperature says otherwise. |
| `tau` | energy | `None` | Fundamental temperature k_B*T directly, as an energy. |

## `[units]`

How to read the numbers in THIS file. Never describes the Hamiltonian file, which carries its own unit in its header.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `energy` | enum | **required** | REQUIRED, deliberately with no default. Governs every key of kind 'energy'. The argparse CLI defaults to eV and RSPt writes Rydberg, so any default here would let two front-ends of the same code disagree by 13.6057x with nothing but a heuristic warning to catch it. A default can be added in a later version; it can never be removed. Choices: `eV`, `Ry`, `Ha`. |
| `temperature` | enum | `'K'` | Whether [temperature].kelvin is Kelvin, or an energy in the unit above. Choices: `K`, `energy`. |

