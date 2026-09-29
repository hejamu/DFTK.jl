# Drude weight, Dreyer et al. Eq. (8):
#   (Ω/π) D_αβ = -∫[d³k] Σ_n f'(ε_nk) v^α_nk v^β_nk
# with ∫[d³k] = Σ_k w_k (weights normalised to 1) and f including the spin factor, so
# (Ω/π) D is in electrons per cell (N_el for free electrons).
#
# The "modified" D̃ replaces the second velocity by the canonical momentum ⟨p_β + k_β⟩
# (kinetic part only). With nonlocal pseudopotentials the NABEC sum rule holds with D̃.
#
# Degenerate manifolds: the diagonal v_nn is gauge dependent inside a degenerate manifold.
# We use the gauge-invariant block form Σ_{n,m∈M} Re(v^α_nm v^β_mn), which equals the
# diagonal form for non-degenerate bands. The same pairs are excluded from the interband
# NABEC sum (see sos.jl), so the sum rule stays exact.

"""
    drude_weight(basis, bands, εF; degeneracy_tol, fp_threshold)

Returns `(; D, Dtilde)`, both 3×3 matrices of ``(Ω/π) D_{αβ}`` in electrons per cell.
"""
function drude_weight(basis::PlaneWaveBasis{T}, bands::BandSet, εF;
                      degeneracy_tol=1e-6, fp_threshold=1e-12) where {T}
    @assert basis.model.n_spin_components == 1 "Spin-polarized case not implemented"
    D  = zeros(T, 3, 3)
    Dt = zeros(T, 3, 3)
    iszero(basis.model.temperature) && return (; D, Dtilde=Dt)
    (; fp) = occupations_and_derivatives(basis, bands, εF)

    for (ik, kpt) in enumerate(basis.kpoints)
        ε  = bands.eigenvalues[ik]
        ψk = bands.ψ[ik]
        v  = VelocityOperator(basis, ik)
        for M in degenerate_manifolds(ε, degeneracy_tol)
            fpM = sum(fp[ik][M]) / length(M)
            abs(fpM) < fp_threshold && continue
            if last(M) == length(ε) && length(ε) < size(ψk, 1)
                @warn "Fermi-surface manifold touches the highest computed band" ik M
            end
            ψM = ψk[:, M]
            V  = [ψM' * apply_velocity(v, α, ψM) for α = 1:3]
            Vt = [ψM' * apply_canonical(v, α, ψM) for α = 1:3]
            for α = 1:3, β = 1:3
                D[α, β]  -= basis.kweights[ik] * fpM * real(tr(V[α] * V[β]))
                Dt[α, β] -= basis.kweights[ik] * fpM * real(tr(V[α] * Vt[β]))
            end
        end
    end
    (; D=DFTK.mpi_sum(D, basis.comm_kpts), Dtilde=DFTK.mpi_sum(Dt, basis.comm_kpts))
end
drude_weight(scfres::NamedTuple, bands=full_bands(scfres); kwargs...) =
    drude_weight(scfres.basis, bands, scfres.εF; kwargs...)
