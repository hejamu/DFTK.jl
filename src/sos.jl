# NABEC by explicit sum over states, Dreyer et al. Eq. (7):
#
#   Z*_{α,κβ} = Z^ion_κ δ_αβ
#     - Im Σ_k w_k Σ_{n≠m} (f_n - f_m) / [(ε_n - ε_m)(ε_n - ε_m + iη)] · v^α_nm H^{τκβ}_mn
#
# v^α = ∂H_k/∂k_α (unscreened, including the nonlocal part), H^τ = screened first-order
# phonon Hamiltonian at q = 0. f includes the spin factor. The result is Z[α, β] per atom:
# α = current (polarization) direction, β = displacement direction.
#
# Evaluation: let O be the bands with f_n > occupation_threshold. Every ordered pair with
# f_n ≠ f_m has n ∈ O or m ∈ O, so with A = ψ† v^α ψ_O and B = ψ† H^τ ψ_O (all bands in the
# rows) the pairs are
#   n ∈ O, m any:    M_nm = v_nm H_mn = conj(A[m,n]) B[m,n]
#   n ∉ O, m ∈ O:    M_nm = A[n,m] conj(B[n,m])
# Pairs with |ε_n - ε_m| < degeneracy_tol are intraband-like and are skipped; they are
# counted in the Drude weight instead (see drude.jl).

"""
    nabec_sos(scfres; bands, phonon, η, degeneracy_tol, occupation_threshold)

Nonadiabatic Born effective charges by explicit sum over states on `bands` (default: full
diagonalization). Returns `(; Z, Zel, Zion, atoms)` with `Z[iatom]` a 3×3 matrix.
Pass `screened=false` to use the bare phonon perturbation (diagnostics only).
"""
function nabec_sos(scfres; bands=full_bands(scfres), phonon=nothing,
                   atoms=1:length(scfres.basis.model.positions),
                   η=0.0, degeneracy_tol=1e-6, occupation_threshold=1e-12,
                   screened=true, tol_response=1e-10)
    basis = scfres.basis
    model = basis.model
    @assert model.n_spin_components == 1 "Spin-polarized case not implemented"
    @assert length(basis.symmetries) == 1 "Use symmetries=false"
    phonon = something(phonon, ScreenedPhonon(scfres; atoms, tol=tol_response))
    @assert phonon.atoms == collect(atoms)
    εF = scfres.εF
    (; f) = occupations_and_derivatives(basis, bands, εF)

    Zel = [zeros(3, 3) for _ in atoms]
    for (ik, kpt) in enumerate(basis.kpoints)
        ψk = bands.ψ[ik]
        ε  = bands.eigenvalues[ik]
        N  = length(ε)
        O  = findall(>(occupation_threshold), f[ik])
        notO = setdiff(1:N, O)
        ψO = ψk[:, O]

        v = VelocityOperator(basis, ik)
        A = [ψk' * apply_velocity(v, α, ψO) for α = 1:3]

        # Complex pair weights W_nm = (f_n - f_m)/((ε_n - ε_m)(ε_n - ε_m + iη))
        pair_weight(n, m) = begin
            Δ = ε[n] - ε[m]
            abs(Δ) < degeneracy_tol && return zero(ComplexF64)
            divided_difference(model, ε[n], ε[m], εF) / (Δ + im * η)
        end
        W1 = [pair_weight(n, m) for m = 1:N, n in O]       # n ∈ O, m any (m ≠ n via Δ)
        W2 = [pair_weight(n, m) for n in notO, m in O]      # n ∉ O, m ∈ O

        for (ia, _) in enumerate(atoms)
            δHψ = apply_δH_cartesian(phonon, ik, ψO, ia; screened)
            for β = 1:3
                B = ψk' * δHψ[β]
                for α = 1:3
                    S1 = sum(imag.(W1 .* conj.(A[α]) .* B))
                    S2 = sum(imag.(W2 .* A[α][notO, :] .* conj.(B[notO, :])))
                    Zel[ia][α, β] -= basis.kweights[ik] * (S1 + S2)
                end
            end
        end
    end
    Zel  = [DFTK.mpi_sum(Z, basis.comm_kpts) for Z in Zel]
    Zion = [charge_ionic(model.atoms[s]) * Matrix(1.0I, 3, 3) for s in atoms]
    (; Z=Zion .+ Zel, Zel, Zion, atoms=collect(atoms))
end
