# NABEC: nonadiabatic Born effective charges in DFTK.jl

Prototype implementation of the nonadiabatic Born effective charges of
Dreyer, Coh & Stengel, PRL **128**, 095901 (2022), on top of DFTK.jl. The aim is a
reference for in-plane atomic polar tensors of water at metal electrodes (Au(111)/water),
where finite-field APTs are ill-defined.

This directory is a standalone Julia project that `dev`s the DFTK checkout it lives in
(`[sources] DFTK = {path = ".."}`); the branch `nabec` is based on DFTK commit `5a94a14`.
DFTK itself is unmodified.

## Contents

| Path | What |
|---|---|
| `src/velocity.jl` | Velocity operator ∂H_k/∂k (kinetic + nonlocal via ForwardDiff on the projectors) |
| `src/drude.jl` | Drude weight D and modified D̃ (canonical momentum) |
| `src/perturbation.jl` | Screened first-order phonon Hamiltonian H^τ at q = 0 (DFTK phonon δH + `solve_ΩplusK_split`) |
| `src/sos.jl` | NABEC by explicit sum over states (full diagonalization, validation only) |
| `src/sternheimer.jl` | NABEC by Sternheimer (P ⊕ Q split), production route |
| `test/` | Phase 1–4 checks (velocity vs finite differences, Drude weight, NaCl ASR, Sternheimer = SOS, H2O vs FD dipole) and system builders |
| `scripts/` | Validation and diagnostic runs (Al, Au, H2O gauge study, slabs); `amam.sh` is the Slurm runner for the TUHH AMAM cluster |
| `NOTES.md` | Conventions, derivations, operational notes |
| `RESULTS.md` | Validation numbers and open issues |
| `results/water_apt_gauge.html` | Interactive comparison of H2O NABEC vs FD APTs |

## Quick start

```julia
using Pkg; Pkg.activate("nabec"); Pkg.instantiate()
using DFTK, NABEC
# scfres from self_consistent_field(...) with symmetries=false
res = nabec_sternheimer(scfres)          # res.Z[i] is the 3×3 tensor of atom i, Z[α, β]:
                                          # α = current/dipole direction, β = displacement
(; D, Dtilde) = drude_weight(scfres, NABEC.converged_bands(scfres.ham, n_bands))
```

## Key findings so far (see RESULTS.md)

- Metal sum rule Σ_κ Z = (Ω/π) D̃ holds to < 1e-3 once the k-mesh resolves the smearing
  (fcc Al, fcc Au).
- H2O: velocity-gauge NABEC = finite-difference dipole APT to < 1e-3 e with PseudoDojo
  pseudopotentials and a box of L ≳ 20 bohr.
- cp2k GTH pseudopotentials are too hard in momentum space for velocity-type quantities
  (no convergence up to 80 Ha); use plane-wave pseudopotentials (PseudoDojo, NLCC off since
  DFTK's phonon response does not include the core density).
- In-plane NABECs in metal slabs need dense in-plane k-meshes; meshes made only of
  time-reversal-invariant points (unshifted 2×2) give D = 0 and meaningless in-plane charges.
