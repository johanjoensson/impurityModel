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
`inverse length` are left alone, which is why `energy_cut` (a multiple of k_B*T, despite
the name) and the NIXS `q` cannot be scaled by mistake. It never describes the
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

**Angular momenta.** `l` is unrestricted here: the format can describe any core/valence
pair. What the current solver can *run* is checked separately, and an unsupported
combination exits saying so and naming what would have to change -- as opposed to a
combination that is wrong at any generality (a dipole transition with
|l_core - l_valence| != 1 is zero by selection rule), which is reported as invalid input.

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

Build from the impurity / hybridization / bath blocks, H = [[H_imp, V^dag], [V, H_bath]].

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `h_imp` | matrix | **required** | Effective impurity block (n_imp, n_imp). |
| `v` | matrix | **required** | Impurity-bath hopping (n_bath, n_imp). |
| `h_bath` | matrix | **required** | Bath block (n_bath, n_bath). |

## `[hamiltonian.crystal_field]`

Build the Hamiltonian from crystal-field parameters. ALL TEN are required: the underlying reader fills each absent key from a hard-coded Ni-in-NiO value, so the shipped CoO/FeO/MnO files (which set six) silently run with Ni's conduction bath.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `e_imp` | energy | **required** | Average valence-shell on-site energy. |
| `e_deltaO_imp` | energy | **required** | Cubic (10Dq) splitting of the valence shell. |
| `e_val_eg` | energy | **required** | Valence bath level coupled to the eg orbitals. |
| `e_val_t2g` | energy | **required** | Valence bath level coupled to the t2g orbitals. |
| `e_con_eg` | energy | **required** | Conduction bath level coupled to the eg orbitals. |
| `e_con_t2g` | energy | **required** | Conduction bath level coupled to the t2g orbitals. |
| `v_val_eg` | energy | **required** | Valence hybridization with the eg orbitals. |
| `v_val_t2g` | energy | **required** | Valence hybridization with the t2g orbitals. |
| `v_con_eg` | energy | **required** | Conduction hybridization with the eg orbitals. |
| `v_con_t2g` | energy | **required** | Conduction hybridization with the t2g orbitals. |
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

## `[hamiltonian.matrix]`

Build from the full one-particle solver matrix, impurity block first.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `h` | matrix | **required** | Full (n, n) one-particle Hamiltonian. |
| `n_impurity_orbitals` | count | **required** | Leading impurity block dimension. |

## `[interaction.none]`

No interaction: a non-interacting reference calculation.

*No declared keys.*

## `[interaction.slater]`

Slater-Condon parameters. Array lengths are DERIVED from the shells' angular momenta (2*l_v+1, 2*l_c+1, 2*l_c+1, 2*l_c+2) and checked, rather than restated as l_core / l_valence keys -- one source of truth per angular momentum.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `F_vv` | energy list | **required** | Valence-valence F^k (was Fdd). Length 2*l_v + 1. |
| `F_cc` | energy list | `None` | Core-core F^k (was Fpp). Length 2*l_c + 1. |
| `F_cv` | energy list | `None` | Core-valence direct F^k (was Fpd). Length 2*l_c + 1. |
| `G_cv` | energy list | `None` | Core-valence exchange G^k (was Gpd). Length 2*l_c + 2. |

## `[interaction.u4_file]`

Read the four-index Coulomb tensor from a file. Out-of-line only: nobody hand-writes n_imp^4 numbers, and the RSPt index convention must be named at the reference site.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `path` | path | **required** | A .npy holding the rank-4 tensor in RSPt convention. |

## `[many_body_basis]`

How the many-body determinant basis is built. Named for the determinant basis specifically: 'basis' alone means both the single-particle orbital basis (a .h0 header declares one) and this, and both appear in one input file.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `truncation_threshold` | auto/count | `'auto'` | Cap on determinants per basis. 'auto' derives it from available per-rank memory at the (collective) call site; 'none' disables capping. The two are NOT interchangeable even though the underlying code currently collapses both to infinity in one place. Choices: `auto`, `none`. |
| `excitation_budget` | auto/count | `'auto'` | Maximum total bath excitations per determinant. 'auto' takes the solver's measured-lossless default; 'none' disables it. Prefer omitting to writing the number: the default is documented as the tightest MEASURED value and is expected to be re-measured, so a copy here would freeze a stale one. Choices: `auto`, `none`. |
| `chain_restrict` | bool | `True` | Apply chain occupation restrictions. |
| `spin_flip_dj` | bool | `False` | Generate spin-flipped determinants. |
| `occ_cutoff` | dimensionless | `None` | Occupation cutoff deciding filled/partial/empty bath classification, i.e. the variational space -- not cosmetic. Per-calculation default. |
| `slater_weight_min` | dimensionless | `None` | Minimum determinant weight retained. |
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
| `outdir` | path | `'.'` | Directory for the output archive. |
| `verbosity` | count | `0` | 0-3; the CLI's -v/-vv/-vvv overrides this. |

## `[selfenergy]`

Impurity self-energy Sigma(w) / Sigma(i nu) and the impurity Green's function.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `cluster` | string | `'cluster'` | Cluster label used in the output filenames. |
| `output` | path | `None` | Output archive; default selfenergy-<cluster>.h5. |

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
| `l` | count | **required** | Angular momentum. UNRESTRICTED by this schema: the format must be able to express any (core l, valence l) pair before the solver can execute it, or it needs replacing the day the 2p/3d restriction lifts. What the solver can actually do is checked separately -- see inputformat.capabilities. |
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
| `auto_block_structure` | bool | `True` | Derive the block structure and symmetry-adapted solver basis from the hybridization-dressed impurity matrix instead of the hand-coded 2p/3d one. A solver-basis decision (it replaces the Hamiltonian operator the solve runs on), which is why it lives here and not under a spectroscopy table. |

## `[spectroscopy]`

PES / XPS / XAS / RIXS / NIXS. The meshes and the core-hole broadening live HERE, not under a technique, because the code genuinely shares them: one `delta` is both the PES/XPS/XAS lineshape and RIXS's intermediate-state broadening, and NIXS is evaluated on RIXS's energy-loss mesh. Filing either under one technique would mean switching that technique off changed another one.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `w` | mesh | `{'min': -25.0, 'max': 25.0, 'n': 3001}` | PES / XPS / XAS evaluation mesh, relative to E_F. |
| `w_loss` | mesh | `{'min': -2.0, 'max': 12.0, 'n': 4000}` | Energy-loss mesh, shared by RIXS and NIXS. |
| `core_hole_broadening` | energy | `0.2` | HWHM above the real axis. Sets the PES/XPS/XAS lineshape AND the RIXS INTERMEDIATE-state resolvent broadening -- one number, two roles, which is why it is not named per technique. |
| `cluster` | string | `'cluster'` | Label used in the output. |
| `output` | path | `'spectra.h5'` | Output archive, relative to [run].outdir. |

## `[spectroscopy.nixs]`

Non-resonant inelastic x-ray scattering, on the shared w_loss mesh.

| Key | Kind | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | Compute it. Previously implied by supplying a radial file; now explicit, and the radial file is required when this is on. |
| `radial_file` | path | `None` | Two-column radial wavefunction of the correlated orbitals. Its length unit is what makes `q` meaningful -- they are reciprocal. |
| `broadening` | energy | `0.1` | HWHM for NIXS. |
| `q` | vector list | `None` | Momentum transfers, reciprocal to the radial mesh's length unit -- an inverse length, so [units].energy does not touch it. NOTE: a q exactly along z currently yields NaN in the transition operator; use a tilted q until that is fixed. |
| `l_final` | count | `2` | Angular momentum of the final orbitals (was liNIXS). |
| `l_initial` | count | `2` | Angular momentum of the initial orbitals (was ljNIXS). |

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
| `output` | path | `'chi.h5'` | Output archive, relative to [run].outdir. |
| `n_psi_max` | count | `5` | Eigenstates to solve for. Configurable on THIS path only: the spectroscopy driver ignores it and the self-energy driver hardcodes its own count. |
| `energy_cut` | dimensionless | `10.0` | Thermal window in multiples of k_B*T -- a MULTIPLIER, not an energy, despite the name; [units].energy must not touch it. |

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

