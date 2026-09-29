# NOTES: conventions, derivations, deviations

DFTK commit: `5a94a1407fa6fa47033e347bd65d7e3dbf163ac4` (2026-09-24, "Remove XC GPU logic
(#1385)"), branch `nabec-pin` in `~/repos/DFTK.jl` on AMAM, `dev`ed via `[sources]` in
`Project.toml`. Julia 1.13.1 (juliaup, AMAM NFS home).

## Conventions

| Quantity | Convention |
|---|---|
| Units | Hartree atomic units (ħ = m_e = e = 1, lengths in bohr, energies in Ha). |
| Charge sign | Electrons carry charge −1. Z* in units of e. Dipole d = Σ_κ Z^ion_κ R_κ − ∫ r ρ. |
| Z tensor index | `Z[κ][α, β]`: α = current / polarization direction, β = displacement direction (Dreyer's Z^(α)_{κβ}). |
| Occupations | `f` includes the spin factor (2 for `spin_polarization=:none`). `f' = ∂f/∂ε` (negative), also including the factor 2. |
| k-integration | ∫[d³k] = Σ_k w_k with DFTK's `basis.kweights` (sum to 1). All results are per cell. |
| Symmetries | `symmetries=false` required (full k-mesh; tensors are not symmetrized). |
| Positions | DFTK stores reduced positions x (R = L x). Phonon δH in DFTK is ∂/∂x_α. Cartesian: ∂/∂R_β = Σ_α (L⁻¹)_{αβ} ∂/∂x_α (`perturbation.jl`). |
| Velocity | v_α = ∂H_k/∂k_α, Cartesian k, G-set held fixed. Kinetic part (k+G)_α. |
| Canonical momentum | p̃_β = p_β + k_β = kinetic part of the velocity only. |
| Projectors | P[G,i] = e^{−i(k+G)·R_i} f_i(k+G)/√Ω (`build_projection_vectors`); V_nl = P D P†. |

### Nonlocal velocity

⟨k+G|V_nl|k+G'⟩ = Ω⁻¹ Σ_ij D_ij f_i(k+G) f_j*(k+G') e^{−i(G−G')·R}: the k-dependence of
the structure factor cancels, only the form factors depend on k. `nonlocal_projector_derivative`
differentiates the full P with ForwardDiff on a dual k-point coordinate (G set fixed); the
extra −iR_α P terms cancel in (∂P)DP† + PD(∂P)†. At k+G = 0 ForwardDiff through `norm`
gives NaN (0·∞): those rows are replaced by a central finite difference (h = 1e-5).
Validated against finite differences of the full H_k(k±h) matrix: relative error ~1e-10
(Si, Al, non-Γ) and ~7e-12 (H2O at Γ).

`DFTK.build_projection_vectors` fixes the element type from the atomic positions, so
`nonlocal_projectors` is a copy that promotes to the k-point's (dual) type.

Kinetic blow-up (`BlowupCHV`, `BlowupAbinit`) is rejected: the velocity would need the
derivative of the modified dispersion.

## Formulas as implemented

### Drude weight (Dreyer Eq. 8)

(Ω/π) D_αβ = −Σ_k w_k Σ_n f'_nk v^α_nk v^β_nk  (electrons per cell).

D̃: the second velocity replaced by ⟨p̃_β⟩. Degenerate manifolds (|Δε| < `degeneracy_tol`)
use the gauge-invariant block form −f'_M Σ_{n,m∈M} Re(v^α_nm v^β_mn); those intra-manifold
pairs are excluded from the NABEC n≠m sum, so the sum rule stays consistent.

Free-electron check: Σ_k w_k Σ_n (−f') v_x² → (Ω/(2π)³)·2·∫d³k δ(ε−ε_F) k_x² = Ω k_F³/(3π²) = N.

### NABEC (Dreyer Eq. 7)

Z*_{α,κβ} = Z^ion_κ δ_αβ − Im Σ_k w_k Σ_{n≠m} (f_n−f_m)/[(ε_n−ε_m)(ε_n−ε_m+iη)] v^α_nm H^τ_mn

- (f_n−f_m)/(ε_n−ε_m) is evaluated with DFTK's stable `occupation_divided_difference`.
- η is a parameter (default 0, the clean static limit); pairs with |Δε| < `degeneracy_tol`
  are skipped for any η (they belong to the Drude/intraband part).
- H^τ = bare local + bare nonlocal ∂V/∂τ + δV_Hxc from `solve_ΩplusK_split` at q = 0.
  At q = 0 DFTK includes δf and δε_F for metals, i.e. the standard adiabatic static
  screening (Dreyer's prescription). The clean-limit variant without Fermi-surface
  repopulation is **not** implemented yet (open issue).

**Sign check (insulator).** For f_a = 2 (occupied), f_b = 0: the pairs (a,b), (b,a) give
Z^el = −2 Σ_a f_a Im Σ_b v_ab H_ba/(ε_a−ε_b)² = −2 Σ f_a Im⟨∂_k u_a|∂_τ u_a⟩.
With v_ab = i(ε_a−ε_b) r_ab for a localized system: Z^el = −Σ_a f_a 2Re⟨a|r|δψ_a⟩ = −∫ r δρ,
i.e. the electronic dipole derivative with electrons negative. So Eq. 7 as written (with −Im)
has the physical sign in DFTK conventions, with no extra sign flips.

### Acoustic sum rule from translation invariance

A rigid translation of all atoms gives Σ_κ H^{τκβ} = −i[p_β, H] (local, nonlocal and Hxc
alike; exact in the plane-wave basis up to xc-grid egg-box effects). Then

Σ_κ Z^el_{α,κβ} = Σ_{n≠m} (f_n−f_m)/(ε_n−ε_m) Re(v^α_nm p̃^β_mn)   (η → 0)

and with ∂_{k_α}⟨n|p̃^β|n⟩ = δ_αβ + Σ_{m≠n} 2Re(v^α_nm p̃^β_mn)/(ε_n−ε_m):

- insulator: Σ_κ Z_κ = Σ_k w_k Σ_n f_n ∂_{k_α}⟨n|p̃^β|n⟩ − N_el + Σ Z^ion. The first term is
  a BZ integral of a total derivative → 0 **only with converged k-sampling**.
- metal: the Fermi-surface part leaves −Σ f' v^α p̃^β = (Ω/π) D̃_αβ (the sum rule with D̃).

Verified numerically: ‖Σ_κ H^τ − (−i[p,H])‖/‖·‖ ≈ 1e-4 for H2O (screened; bare Σ_κ δV_ext
alone misses ~20%).

### Sternheimer split (`sternheimer.jl`)

For η → 0 and non-degenerate pairs the weight is real, (f_n−f_m)/Δ²; combining (n,m),(m,n):
w_nm Im M_nm + w_mn Im M_mn = 2 w_nm Im M_nm (since M_mn = conj(M_nm)). Split bands into P
(computed, converged) and Q (f = 0):

Z^el = −Σ_k w_k [Σ_{n≠m∈P} (f_n−f_m)/Δ² Im M_nm + 2Σ_{n∈P} f_n Im⟨δ⊥_v u_n|δ⊥_τ u_n⟩],
|δ⊥_X u_n⟩ = −Q(H−ε_n)⁻¹Q X|u_n⟩ = Σ_{m∈Q} |m⟩ X_mn/(ε_n−ε_m).

Check: ⟨δ⊥_v u_n|δ⊥_τ u_n⟩ = Σ_{m∈Q} v_nm H_mn/(ε_n−ε_m)² ✓. `DFTK.sternheimer_solver`
with ψk = all P bands and no extra bands solves exactly this. P must be tightly converged
eigenvectors (a separate non-SCF LOBPCG, `converged_bands`), since the P-internal sum uses
them as exact eigenpairs.

## Findings from validation (see RESULTS.md)

- The velocity-gauge ASR is a BZ integral of ∂_k Tr[P_k p̃]: it converges with k-sampling
  (NaCl: 0.20 → 0.013 from 3³ to 6³), not identically.
- The metal sum rule needs the mesh to resolve the smearing (spacing ≲ T/v_F). At fixed
  smearing it closes with D̃ (Al, T = 0.03, 20³: −0.002; T = 0.1, 8³: +0.003) and misses D by
  ≈0.10 e.
- For a molecule in vacuum the velocity and length gauges disagree beyond k and box
  effects (RESULTS.md, open issue 1).
- Au with 11 valence electrons: use `cp2k.nc.sr.lda.v0_1.semicore.gth` (largecore Au has 1 e⁻).
