# RESULTS: validation of the NABEC prototype

All: LDA, GTH (`cp2k.nc.sr.lda.v0_1.largecore.gth`, no NLCC), `symmetries=false`, Gaussian
smearing for metals, DFTK `5a94a14`. Charges in e. Settings are low (fast tests), so values
are not converged w.r.t. Ecut; comparisons are always between methods at identical settings.

## Phase 1: velocity operator ∂H_k/∂k

| System | Check | Result |
|---|---|---|
| Si (Ecut 10, 2³, ik=2) | ‖v − (H(k+h)−H(k−h))/2h‖/‖·‖, fixed G set | 1.4e-10 (all α) |
| Si | max \|⟨n\|v\|n⟩ − dε/dk\| vs independent DFTK bases at k±h | 1.5e-10 |
| Al (Ecut 10, 2³, ik=2) | matrix FD | 1.8e-10 |
| Al | band velocities | 3.8e-11 |
| H2O (Γ, L=10, Ecut 20) | matrix FD at k+G = 0 rows (FD fallback) | 6.7e-12 |
| Cohen-Bergstresser Si | local-only: v = (k+G)_α exactly, no nonlocal part | pass |

## Phase 2: Drude weight

| System | Settings | (Ω/π)D | (Ω/π)D̃ |
|---|---|---|---|
| empty lattice, 3 e⁻ (fcc) | Ecut 3, T=0.04, 20³ | 3.0006 (exact: 3) | = D |
| same | T=0.02, 12³ / 20³ | 2.945 / 2.965 | |
| same | T=0.01, 24³ | 3.219 | |
| fcc Al (1 atom) | Ecut 10, T=0.03, 20³ | 1.926 | 2.030 |
| fcc Al | Ecut 10, T=0.1, 8³ | 1.867 | 1.970 |

The empty-lattice deviation is pure BZ-sampling error (full diagonalization gives identical
numbers to the SCF bands); it needs mesh spacing ≲ T/v_F. D̃ − D ≈ 0.10 e for Al,
matching the ≈0.1 e quoted by Dreyer et al.

## Phase 3: NABEC by sum over states

### Sum rule in fcc Al (1-atom cell, Ecut 10): Σ_κ Z = (Ω/π) D̃

`results/al_sos_T*.tsv`. Z is isotropic, off-diagonal < 1e-11.

| T (Ha) | k-mesh | Z/atom | (Ω/π)D | (Ω/π)D̃ | ΣZ − D | ΣZ − D̃ |
|---|---|---|---|---|---|---|
| 0.03 | 8³ | 1.8151 | 1.8586 | 1.9616 | −0.044 | −0.147 |
| 0.03 | 10³ | 1.8129 | 1.8464 | 1.9502 | −0.034 | −0.137 |
| 0.03 | 12³ | 2.0386 | 1.8083 | 1.9058 | +0.230 | +0.133 |
| 0.03 | 16³ | 2.0378 | 1.9004 | 2.0037 | +0.137 | +0.034 |
| 0.03 | 20³ | 2.0278 | 1.9256 | 2.0302 | +0.102 | **−0.002** |
| 0.1 | 6³ | 1.4624 | 1.3640 | 1.4411 | +0.098 | +0.021 |
| 0.1 | 8³ | 1.9730 | 1.8671 | 1.9702 | +0.106 | +0.003 |
| 0.1 | 12³ | 2.0706 | 1.9627 | 2.0699 | +0.108 | **+0.0007** |
| 0.1 | 16³ | 2.1027 | 1.9934 | 2.1020 | +0.109 | **+0.0007** |
| 0.1 | 20³ | 2.1111 | 2.0015 | 2.1104 | +0.110 | **+0.0007** |
| 0.01 | 4³ … 16³ | −0.04 … 2.07 | 6.40 … 1.68 | | | not converged |

The sum rule holds with the modified Drude weight D̃ once the mesh resolves the smearing
(spacing ≲ T/v_F), and fails with D by ≈0.10 e, as in the paper. At T = 0.01 Ha even 16³
is far from converged (Z has settled at ≈2.07 at 12³–16³, but D has not). Wannier-type
interpolation or much denser meshes are needed for small smearing, as Dreyer et al. note.
The 4-atom conventional cell at T = 0.01 (2³–6³) is equally unconverged; T = 0.1 runs pending.

### NaCl (Ecut 15, insulator): ASR Σ_κ Z_κ = 0

| k-mesh | Z*(Na) | Z*(Cl) | ASR residual |
|---|---|---|---|
| 3³ | 1.0949 | −1.2927 | 0.198 |
| 4³ | 1.0958 | −1.1426 | 0.047 |
| 6³ | 1.0960 | −1.1093 | 0.013 |

The velocity-gauge ASR is a BZ integral of a total derivative (NOTES.md), so it converges
with k. Z*(Na) ≈ 1.10 is in line with LDA literature (≈1.1).

### H2O: NABEC vs finite-difference dipole APT — **OPEN, does not agree**

See "Open issues" below.

## Phase 4: Sternheimer = SOS

| System | settings | max \|Z_Stn − Z_SOS\| | n_P |
|---|---|---|---|
| NaCl | Ecut 12, 2³ | 6.9e-11 | 12 |
| Al (1 atom) | Ecut 10, 4³, T=0.01 | 3.9e-11 | 11 |
| Al (4 atoms) | Ecut 10, 2³, T=0.01 | 4.6e-10 | 18 |

## Phase 5: Au(111)/water slab

Not started (blocked on the H2O issue, see below). Au with 11 e⁻ is only in the
`semicore` GTH-LDA family (`largecore` Au has Zion = 1); O and H are identical in both.

## Open issues

1. **H2O velocity-gauge NABEC ≠ FD dipole APT.** Max deviation per tensor element:

   | L (bohr) | Ecut | k-mesh | max\|Z_NABEC − Z_FD\| | ASR NABEC | ASR FD |
   |---|---|---|---|---|---|
   | 10 | 20 | Γ | 0.140 | 0.186 | 0.013 |
   | 10 | 20 | 2³ | 0.061 | 0.046 | 0.028 |
   | 10 | 20 | 3³ | 0.062 | 0.048 | 0.029 |
   | 12 | 20 | Γ | 0.067 | 0.044 | 0.004 |
   | 14 | 20 | Γ | 0.064 | 0.047 | 0.002 |
   | 16 | 20 | Γ | 0.064 | 0.048 | 0.002 |
   | 12 | 30 | Γ | 0.031 | 0.034 | 0.003 |
   | 12 | 40 | Γ | 0.028 | 0.040 | 0.002 |

   What is verified:
   - velocity operator vs FD of H_k (incl. Γ): ≤1e-10;
   - screened H^τ: −∫ r δρ from DFTK's δρ equals the FD dipole APT to 5 digits;
   - translation identity Σ_κ H^τ = −i[p,H] (1e-4); evaluating the ASR with the ideal
     −i[p̃,H] instead of Σ_κ H^τ gives the same violation to 3 decimals
     (L=12, Ecut 20, Γ: +0.044, +0.016, −0.013 vs +0.044, +0.015, −0.014);
   - Sternheimer = SOS.

   **Localization** (`scripts/diag_water_gauge.jl`, L=12, Ecut 20, Γ, full diag, same H^τ):
   length-gauge SOS (r_nm) = −∫ r δρ exactly; velocity-gauge SOS (v_nm) differs.

   | atom | Z^el_xx vel / len | Z^el_yy vel / len | Z^el_zz vel / len |
   |---|---|---|---|
   | O  | −6.4388 / −6.5061 | −6.2726 / −6.3219 | −6.6033 / −6.6114 |
   | H  | −0.7584 / −0.7448 | −0.8558 / −0.8375 | −0.7049 / −0.6931 |

   So the whole discrepancy is ⟨m|v|n⟩ ≠ i(ε_m−ε_n)⟨m|r|n⟩ in the plane-wave
   discretization of a molecule in vacuum. The Γ/L=10 part is box dispersion (removed by
   k-sampling or larger L); the remainder is independent of L (12–16) and k, and only
   partly reduced by Ecut (0.067 → 0.031 → 0.028 for 20/30/40 Ha, ASR not converging).
   Al and NaCl (no vacuum) do not show it: the Al sum rule closes to 0.003.

   Hypothesis (not confirmed): the gauge identity needs H|m⟩ to stay in the basis. The
   LDA V_xc on the grid in the vacuum region (ρ^{1/3} of density tails) has Fourier
   content up to the grid limit at any Ecut, which would explain the lack of Ecut
   convergence and the absence in bulk. A Hartree-only control was inconclusive (the
   xc-free water model is not a sensible bound insulator). Tests that would settle it:
   Ecut 60–80; a density floor/cutoff in V_xc; a local-only pseudopotential for O
   (removes the nonlocal commutator as a suspect).

   **This matters for Phase 5:** the Au/water slab also has vacuum, and in-plane NABECs
   need the velocity gauge.

2. The bucketed ‖v − iΔr‖ printout in `diag_water_gauge.jl` has the sign of iΔr flipped
   (ratios ≈2 mean v ≈ −rhs); the Z comparison in the same script is unaffected.

3. Clean-limit H^τ (without Fermi-surface repopulation δf, δε_F) is not implemented;
   DFTK's q = 0 response always includes it (handoff §5).

4. Finite ω + iη is a parameter of `nabec_sos` (η), not yet of `nabec_sternheimer`
   (η = 0 only).

5. Traveling-pseudopotential correction (arXiv:2503.18811) not implemented. The sum rule
   closes with D̃ as expected, so the paper's convention is reproduced.
