# The `.h0` file format (v1)

The non-interacting impurity Hamiltonian as written by upstream tooling (`rspt2spectra`'s
`build_h0`) and read by `impurityModel`. This document is the **single source of truth**: the
two implementations live in separate repositories with no shared dependency, so nothing but
this spec and the committed fixtures keeps them in agreement.

Read by `impurityModel selfenergy` and `impurityModel susceptibility` (see the format ×
sub-command table in [`user_guide.md`](user_guide.md)). **Not** read by `impurityModel
spectra`, which needs the labelled `(l,s,m)` formats — see [Scope](#scope).

## Why the format exists

The predecessor (`<cluster>_h0_op.dict`) was a bare list of `i j re im` lines. It recorded no
units, no energy reference, no basis, no impurity/bath boundary and no version, and its
producer discarded the valence/conduction classification it had just computed. Every one of
those omissions is a channel for a silently wrong Hamiltonian, and several of them were in
fact wrong. The format below makes each one explicit and machine-checkable.

## Layout

```
# impurityModel-h0 v1
{"version": 1, "unit": "eV", "n_orb": 106, ...}
--
0 0 -4.126921785541874 0.0
0 20 0.130836789023578 0.09282875396326
```

1. **Magic line** — must match `^#\s*impurityModel-h0\s+v(\d+)\s*$`.
2. **Header** — one JSON object. Written on a single line by `json.dumps`; readers must not
   assume that (see [Parsing](#parsing)).
3. **`--`** — a human-facing separator. Optional and *not* load-bearing; the header's extent
   comes from JSON's own grammar.
4. **Terms** — one `i j re im` line per stored matrix element, whitespace-separated.

### Discriminating the legacy format

**The magic line, not the file extension, decides.** A file whose first non-blank line does
not match the pattern is the legacy bare-integer form and must be routed to
`h0_format.read_legacy_flat_h0`. Extension-based dispatch is not sufficient: legacy `.dict`
files renamed to `.dat` (to get past an older extension check) are common in the wild, and a
legacy file renamed to `.h0` must still be diagnosed correctly rather than mis-parsed.

## Header keys

| Key | Meaning |
| --- | --- |
| `version` | Format version. A reader raises on a version above its own. |
| `required_features` | See [Forward compatibility](#forward-compatibility). |
| `unit` | Energy unit of every amplitude. `"eV"` or `"Ry"`. **Mandatory.** |
| `energy_reference` | `"fermi"` (E_F = 0) or `"absolute"`. **Mandatory** — see [Energy zero](#energy-zero). |
| `fermi_energy` | The E_F that was subtracted, in `unit`. Provenance for the above. |
| `n_orb` | Total spin-orbitals. Authoritative; a term index `>= n_orb` is an error. |
| `index_convention` | `"impurity-block-first"`: impurity occupies `0 .. n_imp-1`, bath after. |
| `spin_ordering` | `"down_first"` / `"up_first"` / `"interleaved"` / `"unknown"`. |
| `shell_layout` | `"single"` or `"multi"`. |
| `storage` | `"full"`: both triangles present, with matching key sets. |
| `impurity_orbitals` | `{group: [indices]}`. A map, so multi-shell is representable. |
| `impurity_l` | Angular momentum of the correlated shell (single-shell files). |
| `valence_bath`, `conduction_bath` | Bath classification. Advisory — see [Bath classification](#bath-classification). |
| `basis` | `"spherical"`, `"cubic"` or `"unknown"` — the impurity orbital basis. |
| `rot_to_spherical` | `n_imp × n_imp`, complex as `[re, im]` pairs. Used for L/S/J reporting. |
| `contains_soc` | Whether spin–orbit coupling is already in the amplitudes. |
| `source_provenance` | Where the local Hamiltonian came from (DFT run vs DMFT iteration N). |
| `interaction` | Optional, e.g. `{"kind": "slater", "l": 2, "F": [...], "xi": 0.0}`. |
| `drop_tolerance` | The relative threshold below which elements were discarded. |
| `bath_geometry` | `"star"`, `"chain"`, `"linked_chain"`, `"peeled_linked_chain"`. |
| `producer` | Versions of all three packages plus a git describe. |

Unknown keys are ignored.

### Forward compatibility

`required_features` lists every header key whose *meaning* the reader must understand to
interpret the term list correctly. A reader that does not recognise an entry raises, printing
producer and consumer versions side by side and pointing here.

A bare `version` integer is not enough. If a later producer adds
`"spin_ordering": "up_first"` and an older reader simply ignores unknown keys, it produces a
Hamiltonian with the magnetization silently flipped. Listing `spin_ordering` in
`required_features` turns that into a loud failure. `unit`, `energy_reference`,
`index_convention`, `spin_ordering` and `storage` are all required features.

For the same reason these are **separate** keys rather than one `index_convention` string:
block ordering, spin ordering and shell layout are independent facts, and each must be
checkable on its own.

## Energy zero

`energy_reference` must be `"fermi"`: the impurity block and the bath block must share a
single energy zero, with E_F at 0.

This is the format's most important invariant, because the predecessor violated it. RSPt's
printed `Local hamiltonian` is an *absolute* energy while the hybridization mesh has E_F = 0,
and `build_h0` never subtracted E_F — so the impurity level sat roughly 8 eV *above the
entire bath*:

| workload | E_F (Ry) | bath (eV) | E_imp as written (eV) | E_imp − E_F (eV) |
| --- | --- | --- | --- | --- |
| NiO | 0.67596 | −6.68 … −1.85 (O 2p) | +8.24 … +8.38 | −0.95 … −0.82 |
| BCC Fe | 0.73692 | −3.97 … −0.99 | +7.70 … +8.52 | −2.32 … −1.51 |
| Nd | 0.41527 | −0.06 … +0.89 | +6.03 … +6.05 | +0.38 … +0.40 (empty 4f) |

`symmetries.classify_bath_occupation` documents the Fermi-level-zero convention as "the same
convention used by the tooling that assembled h0". The bath honoured it; the impurity did not.

**Writer invariant:** `min(diag H_bath) <= mean(diag H_imp) <= max(diag H_bath)`. Refuse to
write otherwise.

## Numeric contract

**Floats are round-trip exact.** Amplitudes are written with `repr(float)` (equivalently
`%.17g`). Fixed-point `%.15f` is *not* acceptable: it is an absolute 1e-15 grid, which is the
wrong shape for float64's relative resolution, and is lossless only for `|x| >= 8` — i.e.
nowhere in these Hamiltonians. In the legacy files it collapsed a median of 36 distinct
float64 values onto each written string, wrote a genuine `6.599434672757963e-12` f-shell
element with four significant digits, and **annihilated 16 terms outright** across four
workloads. `json.dumps` already uses `repr` for floats, so header and body share one rule.

Non-finite values are forbidden. Write with `allow_nan=False` and check
`np.isfinite(...).all()`; `NaN`/`Infinity` are invalid JSON per RFC 8259, and the term parser
rejects them too.

**Hermiticity is the writer's responsibility.** The writer emits `H := (H + H†)/2` — bitwise
Hermitian in float64, since complex addition commutes and `*0.5` is exact — and forces the
diagonal real. Both triangles are stored and the key sets must match.

This is not pedantry. `assemble_h0`'s impurity block is a similarity transform, so it is not
bitwise Hermitian, and the old elementwise `abs(x) > 0` filter therefore kept `(1,0)` while
dropping `(0,1)`. One shipped file (`NiO/ED/Ni_h0_op.dict`) already fails an exact hermiticity
check for exactly that reason. Note also that `%.15f` was accidentally *masking* the problem
by rounding both triangles to the same decimal string, so the symmetrization must land
together with the switch to exact floats, not after it.

**Dropping small elements is pairwise and scale-relative:**

```
drop the pair (i,j),(j,i)  iff  max(|H[i,j]|, |H[j,i]|) <= drop_tolerance * max|H|
```

applied *before* formatting, with `drop_tolerance` defaulting to `1e-12` (the code's own noise
floor — cf. `BREAKDOWN_TOL`, `rotate_hamiltonian(tol=1e-12)`). Pairwise is what keeps the key
sets symmetric. A flat *absolute* cut is wrong: an f-shell workload has a continuous magnitude
distribution from 6.6e-12 up to 4.4e-1 with no gap, so an absolute 1e-10 would delete real
physics.

**A repeated `(i,j)` is an error**, not an accumulation. Floating addition is not associative
(measured spread across orderings of five-term groups: 2.7e-14), and duplicates listed in
different orders for `(i,j)` and `(j,i)` would break hermiticity non-deterministically.

## Parsing

1. Read bytes; strip a leading UTF-8 BOM; decode UTF-8.
2. `splitlines()` — handles LF, CRLF, CR and a missing final newline in one call.
3. Match the magic line on the first non-blank line; if it does not match, treat the file as
   legacy (see above).
4. Take the header's extent from JSON's own grammar:
   `json.JSONDecoder().raw_decode(text, offset_of_first_brace)`. **Do not scan for a line
   equal to `--`** — that breaks on CRLF, on trailing whitespace, and on `--` appearing
   inside a JSON string value.
5. In the remainder, skip blank lines and lines starting with `#`. Every other line must
   `split()` into exactly four fields.

Every error carries the path, the **1-based line number**, `repr(line)` and what was expected.
The leaf parsers do not know the filename, so they raise a bare `ValueError` and the top-level
reader re-raises with context.

## Bath classification

`valence_bath` / `conduction_bath` are **advisory**. The self-energy path re-derives its own
split (`solver_basis` → `symmetries.classify_bath_occupation`) using the same `h[o,o] < 0`
criterion the producer used, so a reader should *compare* rather than trust: a disagreement
means a units error, a spin swap or a Fermi-level offset, and is worth raising on.

For non-star bath geometries the producer should omit them. The classification is derived from
the star Hamiltonian, but the written matrix is the requested geometry, where each bath site is
a Lanczos combination of star modes — so a per-index label from the star is an index into a
different basis, not merely an approximation.

Note the fields discriminate little in practice: with `valence_bath_only=True` the conduction
list is empty for every 3d workload.

## Interaction

`interaction` is optional, but its absence must not be silent. A model built with `u4 = None`
runs a **non-interacting** ED solve to completion and prints a plausible-looking Σ(ω); readers
must therefore raise unless the interaction is supplied either by the header or by the caller.

When present it is the **atomic Slater–Condon** form, reconstructed via `model.atomic_u4`.
That is *not* the same object as the screened rank-4 `u4` the in-memory RSPt interface receives
over CFFI. Do not compare self-energies across the two paths and attribute the difference to
the file format.

`xi` and `contains_soc` interact: if the amplitudes already contain spin–orbit coupling,
dressing them again with `xi` double-counts, and the reader must refuse a non-zero `xi`.

## Scope

`impurity_orbitals` is a group → indices map, so a multi-shell file is representable. The
current producer does not emit one: RSPt writes one hybridization file per orbital group, so
`build_h0` output is one shell per file even when the `green.inp` cluster lists several.

The `spectra` sub-command does **not** read this format. Its Hamiltonian builder constructs
spin–orbit coupling, the Slater–Condon interaction and the MLFT double counting in `(l,s,m)`
labels and interleaves shells under `c2i`, so a flat-index Hamiltonian would need an explicit
permutation to be correct there.

## Legacy format

Bare `i j re im` lines, no header, both triangles, `%.15f` amplitudes, flat indices. Read with
`h0_format.read_legacy_flat_h0(path, n_imp)`; the reader symmetrizes on load and emits a
`DeprecationWarning`.

`n_imp` cannot be inferred from the file, but a wrong value **can** be caught. The impurity
block is dense and hybridizes to every bath orbital, while the bath block is diagonal (star) or
tridiagonal (chain), so

```
k* = min{ k : bandwidth(sparsity[k:, k:]) <= 1 }
```

recovers it. Over the nine stored legacy files this is correct 9/9, with margins of 7–154
against the threshold, and it finds an f-shell `n_imp = 14` unaided. Use it as a **check** that
raises and suggests `k*`, never as a silent inference — and note the bandwidth assumption is
verified for star and chain geometries only.

## Cross-checking the two implementations

These need `rspt2spectra`, `impurityModel` and the workload data present at once, which no CI
has. They are a manual procedure, not a test.

1. In an RSPt directory, run the `build_h0` pipeline, keep the in-memory `H` from
   `assemble_h0`, write the `.h0`, read it back, densify and compare elementwise.
2. **Physical placement.** Assert the impurity diagonal lies within the bath's energy range,
   and that `E_imp − E_F` has the expected sign and magnitude (a few eV negative for a
   partially filled 3d shell, slightly positive for an empty 4f).
3. **Hybridization reconstruction.** Form `Δ(z) = V†(z − H_bath)⁻¹V` from the parsed `h0` and
   compare against RSPt's input `real-/imag-hyb-<cluster>.dat`. This catches index order, spin
   order and a misplaced impurity/bath boundary — but it is **structurally blind to the energy
   zero and the unit**, since it touches only the bath and hopping blocks. Step 2 is what
   covers those, and must not be dropped.
4. A collapsed basis is silent. Confirm the impurity occupation is what you expect (for NiO,
   ~8.6 — neither 10 nor ~0) and the spectrum non-zero before trusting anything.
