# Companion Plan: Non-Abelian Symmetry Detection & Casimir Reconstruction

**Status:** Companion to `symmetry_implementation_plan.md`. Phase 2 of that plan
discovers the one-body symmetry algebra and picks an *abelian* (Cartan) subalgebra
for sector labels. This plan covers what to do with the **rest** of the discovered
algebra — the non-abelian part — and how to reconstruct Casimir operators for exact
multiplet labeling.

> **Execution conventions & suitability:** see "Implementation conventions for
> weak-model execution" in `README.md`. **This entire plan is [strong-model] research
> work** — Lie-algebra structure-constant extraction and Casimir reconstruction need
> mathematical judgment. It also *depends on* `symmetry_implementation_plan.md` Phase 2
> existing first (it consumes that phase's discovered null space). A small/fast model
> should not attempt it; the value of the detail below is to brief whoever does. The
> formulas (`S_x/S_y/S_z`, `Ĉ = Σ_a O_a·O_a`) are exact and given — implement them as
> written, with `test_casimir_commutes_with_H` as the non-negotiable correctness gate.

---

## Motivation

The constraint `[H, O] = 0` over one-body operators `O` returns the full one-body
symmetry algebra, not just the commuting generators. For an SU(2)-symmetric
Hamiltonian the null space contains **all three** spin generators, because each is
one-body:

```
S_z = ½ Σ_i (n_{i↑} − n_{i↓})
S_x = ½ Σ_i (c†_{i↑} c_{i↓} + c†_{i↓} c_{i↑})
S_y = (1/2i) Σ_i (c†_{i↑} c_{i↓} − c†_{i↓} c_{i↑})
```

The main plan's Phase 2.3 selects the Cartan subalgebra `{N̂, S_z}` for sector labels
(these mutually commute and map to occupation restrictions). But `S_x`, `S_y` are
*also* in the discovered null space — they just don't commute with `S_z`. Their
presence is the signature of a non-abelian symmetry, and from them the Casimir

```
Ŝ² = S_x² + S_y² + S_z²
```

can be reconstructed **as a derived operator** rather than hand-built. The same logic
applies to orbital angular momentum `L̂²` (from one-body `L_x, L_y, L_z` when the
point group supports it) and, with spin-orbit coupling, `Ĵ²`.

This matters because:
- **Multiplet labeling:** abelian quantum numbers `(N, S_z)` cannot distinguish a
  singlet from the `S_z=0` component of a triplet. The Casimir `Ŝ²` can.
- **Degeneracy explanation:** a `(2S+1)`-fold degeneracy in the spectrum is *proven*
  (not just observed) once `Ŝ²` is available and commutes with `H`.
- **Validation:** `[Ŝ², H] = 0` is a strong, automatic correctness check on the whole
  discovery pipeline.

> **Important:** `Ŝ²` itself is a *two-body* operator and is therefore **not** in the
> one-body null space. It is *reconstructed* from the one-body generators that are.
> This is the distinction the main plan's Phase 2 makes — this companion plan handles
> the reconstruction step explicitly.

---

## Scope and non-goals

- **In scope:** detecting that the discovered one-body algebra is non-abelian;
  identifying its structure (closing under commutation → a Lie algebra);
  reconstructing Casimir operators as `ManyBodyOperator`s; using them for multiplet
  labeling and validation.
- **Out of scope (stated as known limitations):**
  - **Anti-unitary symmetries (time reversal).** `[H, O] = 0` over the complex field
    finds only *unitary* symmetries. Time reversal is anti-unitary and produces
    Kramers degeneracy that this method **cannot** detect. Document this; do not
    claim Kramers pairs are discovered.
  - **Block-diagonalization by non-abelian symmetry.** Using the full SU(2) to
    reduce the Hilbert space (Clebsch-Gordan / Wigner-Eckart reduced matrix elements)
    is a much larger effort and is explicitly not attempted here. We use the
    non-abelian symmetry only for *labeling and validation*, while sectorization in
    the main plan stays abelian (occupation restrictions).

---

## Phase A: Detect non-abelian structure

**Location:** `src/impurityModel/ed/symmetries.py`

### A.1. Classify the discovered algebra

- **Action:** After Phase 2.2 returns the one-body null space `{O_a}`, compute all
  pairwise commutators `[O_a, O_b]` (single-particle matrices). The algebra is
  abelian iff all commutators vanish (to the SVD threshold). If any are non-zero, the
  symmetry is non-abelian.
- **Action:** Verify the set closes under commutation: each `[O_a, O_b]` must be a
  linear combination of the `{O_c}` (it is, if `{O_a}` is a complete null-space
  basis). Extract the structure constants `f_{abc}` where `[O_a, O_b] = Σ_c f_{abc}
  O_c`. For SU(2) these reproduce the Levi-Civita symbol up to normalization.
- **Verification:**
  - [ ] `test_detect_nonabelian_su2` — SU(2)-symmetric Hubbard dimer; assert exactly
        three spin generators are found, the algebra is non-abelian, and the structure
        constants match `su(2)` (`[S_a, S_b] = i ε_{abc} S_c`) after normalization.
  - [ ] `test_detect_abelian` — a model with only `N̂` and a crystal-field-split
        orbital symmetry; assert the algebra is abelian (all commutators zero).

### A.2. Reconstruct Casimir operators

- **Action:** Identify simple sub-algebras within the discovered set (e.g., the
  `su(2)` spin triplet). For each, build the Casimir as a two-body `ManyBodyOperator`:
  `Ĉ = Σ_a O_a · O_a` (operator product, yielding two-body terms). Reuse the
  observable-construction machinery from main-plan Phase 1.2 — `Ŝ²` here and the
  hand-built `Ŝ²` there must be **identical**; assert it.
- **Action:** Where both spin and orbital `su(2)` are present, also form
  `Ĵ² = (L̂ + Ŝ)²` from the combined generators.
- **Verification:**
  - [ ] `test_casimir_matches_handbuilt` — reconstructed `Ŝ²` equals the explicitly
        constructed `Ŝ²` from Phase 1.2 to machine precision (operator-coefficient
        comparison).
  - [ ] `test_casimir_commutes_with_H` — `‖[Ĉ, H]‖ < threshold` for every
        reconstructed Casimir (the core correctness gate).

---

## Phase B: Multiplet labeling

**Location:** `src/impurityModel/ed/groundstate.py` / `finite.py`

### B.1. Label degenerate manifolds by Casimir eigenvalue

- **Action:** Given a (possibly degenerate) low-energy manifold from block Lanczos,
  evaluate each reconstructed Casimir **within the manifold** (diagonalize `Ĉ`
  restricted to the degenerate eigenspace — see main-plan Phase 1.5 for why a single
  Lanczos vector is not enough). Map eigenvalues `S(S+1)`, `L(L+1)`, `J(J+1)` back to
  `S`, `L`, `J`.
- **Action:** Tag printed output with `(N, S, L, J)` multiplet labels, finite-T
  weighted where appropriate (reuse `average.py`).
- **Verification:**
  - [ ] `test_multiplet_labeling` — model with known multiplet structure (e.g.,
        atomic `d²` under a cubic field); assert the reported `(S, L, J)` labels match
        the textbook term symbols and that degeneracies equal `(2S+1)(2L+1)` (or
        `2J+1` with SOC).

---

## Acceptance

- [ ] Non-abelian detection, Casimir reconstruction, and multiplet labeling all pass
      serial and under `mpirun -n {2,3,4}`.
- [ ] `[Ĉ, H] = 0` holds for every reconstructed Casimir on every test model.
- [ ] Reconstructed `Ŝ²`/`L̂²`/`Ĵ²` are bit-for-bit identical to the hand-built
      operators in main-plan Phase 1.2 (single source of truth — pick one and have the
      other assert against it).
- [ ] Documentation clearly states the two known limitations (no anti-unitary/Kramers
      detection; no non-abelian block-diagonalization).
