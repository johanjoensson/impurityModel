# Core-level spectroscopy from RSPt: interface contract

This document fixes what impurityModel offers and what RSPt (through `impmod_interface`) has to
hand over to run X-ray spectroscopy on a cluster. Nothing here is implemented on the RSPt side
yet; the impurityModel side is in place (`spectroscopy.py`, `spectra_techniques.py`,
`ImpurityModel.from_blocks`).

## What impurityModel provides

```python
from impurityModel.api import (ImpurityModel, calc_spectroscopy, calc_xas, save_spectra,
                               load_spectroscopy_requests)

model = ImpurityModel.from_blocks(
    H_imp, V, H_bath, dc=dc, u4=u4,
    rot_to_spherical=R,            # (n_imp, sum_a 2(2 l_a + 1)), columns grouped per shell
    shell_ls=(1, 2),               # l of each column block, e.g. 2p core + 3d valence
    valence_shell=1, core_shell=0, # indices into shell_ls
    bath_valence_conduction=(valence_idx, conduction_idx),
)
requests = load_spectroscopy_requests("impurityModel.toml", comm=comm)  # [spectroscopy.*] tables only
result = calc_spectroscopy(model, requests, basis, solver, comm=comm)   # rank 0: dict, others: None
save_spectra(result, "spectra-<label>.h5")
```

* One call solves the ground state once and runs every requested spectrum; `calc_xas`, `calc_xps`,
  `calc_pes`, `calc_rixs` run one. `result` is `{"E", "datasets", "meshes", "ensemble", "gs_info"}`
  on rank 0 and `None` on every other rank; nothing is written back into RSPt memory.
* A block model needs no `(l, s, m)` labels. Shell membership of an impurity orbital is the shell of
  `rot_to_spherical` holding its weight; the bath belongs to the valence shell; the core shell has none.
* The interacting operator is `h0 - dc + U(u4)` (`solver_basis.assemble_interacting_operator`, the same
  sign convention as the self-energy path).
* NIXS and projected XAS/RIXS are not available on a block model (they need radial data / labelled
  indices); they raise before any collective.
* Technique settings (meshes, broadenings, polarizations) come from `impurityModel.toml`
  `[spectroscopy.*]`; RSPt passes the model and the operators only.

## Transition operators

`XASOptions.dipole` / `RIXSOptions.dipole` take a `(3, n_imp, n_imp)` array: the x, y, z dipole
matrices in the model's input basis, rows the valence and columns the core orbitals (the other
blocks are ignored). With `dipole=None` the atomic Gaunt dipole is built in the spherical basis and
rotated, `T_input = R_valence D R_core^†`. Three Cartesian components are what the tensor kernels
(`calc_spectra_tensor`, `calc_tensor_map`) want; polarization is applied afterwards.

Equalities that the tests pin (`test/spectra/test_block_model_spectra.py`): a block model of the
same operator reproduces the labelled spectra; rotating the valence shell leaves the XAS tensor and
the PES trace unchanged; an explicit dipole equal to the atomic one changes nothing; scaling it by
`c` scales the tensor by `|c|^2`; a dipole with valence and core blocks swapped gives a different spectrum.

## What RSPt's `observables` can and cannot do

Read in `rspt/rsptDir/src`: `green_config.F90:82-88,156-167`, `green_init.F90:2300-2336`,
`green_locust.F90:3563-3800, 5296-5406`.

Usable:

* Per cluster, `acluster%observables%obs(:)` is a list of **named one-body matrices `t(msize, msize)`
  in the correlated basis** -- what an index-based layer consumes.
* `Dxyz` (or `Dpm`) makes RSPt build the three dipole matrices through `locust_transition` (LDA
  transition matrix elements projected on the cluster, site-symmetry rotated). With the core set
  and the valence set both in the cluster, the `[valence rows, core columns]` block is the XAS/RIXS
  transition operator in the right basis and local axes.
* `Photon` + `k(3)` + `helicity` are parsed and never read in RSPt; free to forward as default
  polarization metadata (tensors are stored polarization-free anyway).
* `observables-<name>.inp` custom matrices give arbitrary one-body operators: a later route to
  quadrupole edges or generic `chi_AB`.

Limits (all handled by the design above):

* The block is global, not per cluster; names are at most 10 characters, at most 20 entries; there
  are no meshes, broadenings or switches, hence the settings come from `impurityModel.toml`.
* `obs(l)%name` is a display label (`"(Dipole) Dx"`, `"         Dy"`); match on the trailing token
  (`Dx|Dy|Dz|Dp|Dm`).
* The matrix covers the whole cluster (intra-shell blocks included) and is hermitized; only the
  valence-row / core-column block is used. `locust_transition` is `q = 0` only.
* A multi-set cluster (`ncorr >= 2`) is needed, which `green_solver.F90:1411` refuses for the ED hook
  today, together with the set partition (`offset`, `preoffset`, `nspmat`), which the current
  `run_impmod_ed` does not pass (`impmod_interface` has no `shell_ls` yet). RSPt has no core flag:
  the core set must be named.
* First use must compare the RSPt matrix with the atomic (Gaunt) dipole: for a full shell they may
  differ by one radial constant only. `dipole=` accepts either source.

## Proposed C entry point

```c
int run_impmod_cls(char label[18], char solver_param[120],
                   double _Complex *U_mat, double _Complex *hyb, double _Complex *h_dft, double *w,
                   double _Complex *corr_to_spherical,
                   int *offset, int *preoffset, int *shell_l, int n_shells, int nspmat, int core_shell,
                   double _Complex *obs, char *obs_names, int n_obs, double *photon_k, double _Complex helicity,
                   size_t n_orb, size_t n_orb_full, size_t n_w, double eim, double tau, int verbosity,
                   size_t size_real, size_t size_complex);
```

* Array shapes and order follow `run_impmod_ed` (column-major Fortran arrays, `n_orb` impurity
  spin-orbitals over all shells of the cluster, `n_orb_full` including the bath); `obs` is
  `(n_orb, n_orb, n_obs)`, `obs_names` is `n_obs` blank-padded 10-character labels.
* `shell_l[n_shells]` and `offset`/`preoffset` give the shells and their column blocks of
  `corr_to_spherical`; `core_shell` is the (zero-based) shell index of the core set.
* Return code as the existing entry: `error = -1` for a handled failure, `-99` for an unexpected
  one; no partial results are written on error.

Python flow inside `impmod_interface`:

1. fit the bath as on the self-energy path;
2. `ImpurityModel.from_blocks(..., shell_ls, valence_shell, core_shell)`;
3. `load_spectroscopy_requests("impurityModel.toml")`;
4. replace `dipole` of the XAS/RIXS requests with the `Dx/Dy/Dz` blocks from `obs` (`dataclasses.replace`);
5. `calc_spectroscopy` and `save_spectra("spectra-<label>.h5")`.

## RSPt-side prerequisites

* lift the multi-set guard for the ED hook (`green_solver.F90:1411`);
* pass the set partition and the correlated-to-spherical rotation per shell;
* put `Dxyz` in `observables` and name the core set.
