# NABEC by Sternheimer equations (no full diagonalization).
#
# Split the band space into P (n_P explicitly computed, converged eigenpairs; contains all
# bands with f > 0) and its complement Q, where f = 0. With η → 0 and pairs of (n,m),(m,n)
# combined:
#
#   Z^el = -Σ_k w_k [ Σ_{n≠m ∈ P} (f_n-f_m)/(ε_n-ε_m)² Im M_nm
#                     + 2 Σ_{n∈P} f_n Im⟨δ⊥_v u_n | δ⊥_τ u_n⟩ ]
#   |δ⊥_X u_n⟩ = -Q (H - ε_n)^{-1} Q X|u_n⟩
#
# The P-internal part is the sum-over-states kernel restricted to P; the Q part needs one
# Sternheimer solve per perturbation (3 velocities + 3 n_atoms displacements) for the
# occupied bands. Derivation in NOTES.md, "Sternheimer split".

"""
Tightly converged lowest `n_bands` eigenpairs of the SCF Hamiltonian (non-self-consistent).
"""
function converged_bands(ham, n_bands; tol=1e-10, n_buffer=4)
    res = DFTK.diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, n_bands + n_buffer;
                                       tol, miniter=1, maxiter=400)
    res.converged || @warn "Band diagonalization not converged"
    BandSet([ψk[:, 1:n_bands] for ψk in res.X], [λk[1:n_bands] for λk in res.λ])
end

"""
Contribution of the ordered pairs (n, m) with n ∈ O or m ∈ O, both in the band set of
`ψ` (columns) — the sum-over-states kernel of `sos.jl`. `A`, `B` are ψ' X ψ_O.
"""
function pair_sum(model, ε, O, εF, A, B; η, degeneracy_tol)
    N = length(ε)
    notO = setdiff(1:N, O)
    pair_weight(n, m) = begin
        Δ = ε[n] - ε[m]
        abs(Δ) < degeneracy_tol && return zero(ComplexF64)
        divided_difference(model, ε[n], ε[m], εF) / (Δ + im * η)
    end
    W1 = [pair_weight(n, m) for m = 1:N, n in O]
    W2 = [pair_weight(n, m) for n in notO, m in O]
    sum(imag.(W1 .* conj.(A) .* B)) + sum(imag.(W2 .* A[notO, :] .* conj.(B[notO, :])))
end

"""
    nabec_sternheimer(scfres; n_bands, phonon, tol_sternheimer, ...)

NABEC with the split P ⊕ Q formula. `n_bands` sets the size of P (default: occupied bands
plus a margin). Returns the same named tuple as `nabec_sos`.
"""
function nabec_sternheimer(scfres; n_bands=nothing, phonon=nothing,
                           atoms=1:length(scfres.basis.model.positions),
                           degeneracy_tol=1e-6, occupation_threshold=1e-12,
                           tol_sternheimer=1e-10, tol_response=1e-10, tol_bands=1e-10,
                           maxiter_sternheimer=500)
    basis = scfres.basis
    model = basis.model
    @assert model.n_spin_components == 1 "Spin-polarized case not implemented"
    @assert length(basis.symmetries) == 1 "Use symmetries=false"
    phonon = something(phonon, ScreenedPhonon(scfres; atoms, tol=tol_response))
    @assert phonon.atoms == collect(atoms)
    εF = scfres.εF

    n_occ_max = maximum(count(>(occupation_threshold), fk) for fk in scfres.occupation)
    n_bands = something(n_bands, n_occ_max + max(8, n_occ_max ÷ 2))
    bands = converged_bands(scfres.ham, n_bands; tol=tol_bands)
    (; f) = occupations_and_derivatives(basis, bands, εF)

    Zel = [zeros(3, 3) for _ in atoms]
    for (ik, kpt) in enumerate(basis.kpoints)
        ψP = bands.ψ[ik]
        ε  = bands.eigenvalues[ik]
        O  = findall(>(occupation_threshold), f[ik])
        @assert maximum(O) < n_bands "P must contain empty bands above all occupied ones"
        ψO = ψP[:, O]
        fO = f[ik][O]
        Hk = scfres.ham.blocks[ik]

        # All of P is projected out; no Schur-complement extra bands (the solver's
        # defaults for those are sized like ψk and would not match ε[O]).
        no_extra = zeros(eltype(ψP), size(ψP, 1), 0)
        δ⊥(rhs) = DFTK.sternheimer_solver(Hk, ψP, ε[O], rhs; tol=tol_sternheimer,
                                          ψk_extra=no_extra, Hψk_extra=no_extra,
                                          εk_extra=zeros(real(eltype(ψP)), 0),
                                          maxiter=maxiter_sternheimer).δψk

        v    = VelocityOperator(basis, ik)
        vψO  = [apply_velocity(v, α, ψO) for α = 1:3]
        A    = [ψP' * vψO[α] for α = 1:3]
        δ⊥v  = [δ⊥(vψO[α]) for α = 1:3]

        for (ia, _) in enumerate(atoms)
            δHψ = apply_δH_cartesian(phonon, ik, ψO, ia)
            for β = 1:3
                B   = ψP' * δHψ[β]
                δ⊥τ = δ⊥(δHψ[β])
                for α = 1:3
                    SP = pair_sum(model, ε, O, εF, A[α], B; η=0.0, degeneracy_tol)
                    SQ = 2 * sum(fO[j] * imag(dot(δ⊥v[α][:, j], δ⊥τ[:, j]))
                                 for j in eachindex(O))
                    Zel[ia][α, β] -= basis.kweights[ik] * (SP + SQ)
                end
            end
        end
    end
    Zel  = [DFTK.mpi_sum(Z, basis.comm_kpts) for Z in Zel]
    Zion = [charge_ionic(model.atoms[s]) * Matrix(1.0I, 3, 3) for s in atoms]
    (; Z=Zion .+ Zel, Zel, Zion, atoms=collect(atoms), n_bands)
end
