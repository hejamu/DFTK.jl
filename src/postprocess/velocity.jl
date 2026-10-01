# Velocity operator and Drude weight.
#
# The velocity operator of k-point block ik is v_α = ∂H_k/∂k_α (Cartesian α) at fixed
# G-vector set, assembled from the `∂kH_operators` of all terms (kinetic and nonlocal;
# local, Hartree and xc terms have no explicit k-dependence). Its kinetic part alone,
# (k+G)_α, is the canonical momentum p̃_α = p_α + k_α.

"""
    velocity_operators(basis, ik; canonical=false)

Return a 3-tuple of functions `ψk -> v_α ψk` applying the velocity operator
``v_α = ∂H_k/∂k_α`` (Cartesian α, Hartree atomic units) to orbitals of k-point `ik`.
With `canonical=true` only the kinetic part (the canonical momentum ``p_α + k_α``) is
applied.
"""
function velocity_operators(basis::PlaneWaveBasis, ik::Integer; canonical=false)
    terms = canonical ? [t for t in basis.terms if t isa TermKinetic] : basis.terms
    ops = filter(!isnothing, [∂kH_operators(term, basis, ik) for term in terms])
    ntuple(α -> (ψk -> sum(op[α] * ψk for op in ops)), 3)
end

# Group consecutive eigenvalues closer than tol into degenerate manifolds.
function degenerate_manifolds(ε, tol)
    manifolds = UnitRange{Int}[]
    start = 1
    for n = 2:length(ε)
        if ε[n] - ε[n-1] ≥ tol
            push!(manifolds, start:n-1)
            start = n
        end
    end
    push!(manifolds, start:length(ε))
end

is_time_reversal_invariant(k; atol=1e-10) = all(x -> isapprox(2x, round(2x); atol), k)

@doc raw"""
    compute_drude_weight(basis, ψ, eigenvalues, εF; degeneracy_tol=1e-6)
    compute_drude_weight(scfres; kwargs...)

Drude weight in the form ``(Ω/π) D_{αβ}`` (electrons per unit cell; equal to the number of
electrons for free electrons), Eq. (8) of Dreyer, Coh, Stengel, PRL **128**, 095901 (2022):
```math
\frac{Ω}{π} D_{αβ} = -\sum_k w_k \sum_n f'(ε_{nk})\, v^α_{nk} v^β_{nk},
```
with ``f'`` the derivative of the occupation (including the spin factor). Returns
`(; D, Dtilde)`, where `Dtilde` uses the canonical momentum ``⟨p̃_β⟩`` in place of the
second velocity: with nonlocal pseudopotentials the sum rule of the nonadiabatic Born
effective charges (see [`compute_nabec`](@ref)) holds for `Dtilde`.

Inside degenerate manifolds the gauge-invariant block form ``\sum_{n,m∈M} v^α_{nm} v^β_{mn}``
is used. The bands in `ψ` must be converged and include all bands with non-negligible
``f'``. Requires `symmetries=false` (the tensor is not symmetrized).
"""
function compute_drude_weight(basis::PlaneWaveBasis{T}, ψ, eigenvalues, εF;
                              degeneracy_tol=1e-6, fp_threshold=1e-12) where {T}
    model = basis.model
    @assert model.n_spin_components == 1 "Spin-polarized case not implemented"
    @assert length(basis.symmetries) == 1 "Drude weight requires symmetries=false"
    D  = zeros(T, 3, 3)
    Dt = zeros(T, 3, 3)
    iszero(model.temperature) && return (; D, Dtilde=Dt)
    if all(kpt -> is_time_reversal_invariant(kpt.coordinate), basis.kpoints)
        @warn("All k-points are time-reversal invariant: band velocities vanish there " *
              "and the Drude weight is zero by symmetry.")
    end

    filled = filled_occupation(model)
    temperature = model.temperature
    for (ik, kpt) in enumerate(basis.kpoints)
        εk = eigenvalues[ik]
        fp = filled .* Smearing.occupation_derivative.(model.smearing,
                                                        (εk .- εF) ./ temperature) ./ temperature
        v  = velocity_operators(basis, ik)
        p̃  = velocity_operators(basis, ik; canonical=true)
        for M in degenerate_manifolds(εk, degeneracy_tol)
            fpM = sum(fp[M]) / length(M)
            abs(fpM) < fp_threshold && continue
            if last(M) == length(εk)
                @warn "Fermi-surface manifold touches the highest computed band" ik M
            end
            ψM = ψ[ik][:, M]
            V  = [ψM' * v[α](ψM) for α = 1:3]
            Vt = [ψM' * p̃[α](ψM) for α = 1:3]
            for α = 1:3, β = 1:3
                D[α, β]  -= basis.kweights[ik] * fpM * real(tr(V[α] * V[β]))
                Dt[α, β] -= basis.kweights[ik] * fpM * real(tr(V[α] * Vt[β]))
            end
        end
    end
    (; D=mpi_sum(D, basis.comm_kpts), Dtilde=mpi_sum(Dt, basis.comm_kpts))
end
function compute_drude_weight(scfres::NamedTuple; kwargs...)
    compute_drude_weight(scfres.basis, scfres.ψ, scfres.eigenvalues, scfres.εF; kwargs...)
end
