# Band sets on which the sum-over-states quantities are evaluated.

"""
Eigenpairs per k-point: `ψ[ik]` (n_G × n_bands) and `eigenvalues[ik]`, sorted ascending.
"""
struct BandSet{Tψ, Tε}
    ψ::Vector{Tψ}
    eigenvalues::Vector{Tε}
end

"""
All eigenpairs of the converged SCF Hamiltonian, from a full dense diagonalization of each
k-block. This makes the n ≠ m sums exact within the plane-wave basis. Only for small cells
and low cutoffs.
"""
function full_bands(ham::Hamiltonian)
    ψ = Matrix{ComplexF64}[]
    ε = Vector{Float64}[]
    for block in ham.blocks
        F = eigen(Hermitian(Matrix(Array(block))))
        push!(ψ, F.vectors)
        push!(ε, F.values)
    end
    BandSet(ψ, ε)
end
full_bands(scfres::NamedTuple) = full_bands(scfres.ham)

"""
The bands stored in the SCF result (occupied plus the extra bands DFTK keeps).
"""
scf_bands(scfres) = BandSet(scfres.ψ, scfres.eigenvalues)

"""
Occupations (including the spin factor) and ``∂f/∂ε`` at the Fermi level `εF`.
"""
function occupations_and_derivatives(basis, bands::BandSet, εF)
    model = basis.model
    T = model.temperature
    filled = DFTK.filled_occupation(model)
    f = [filled .* DFTK.Smearing.occupation.(model.smearing, (εk .- εF) ./ T)
         for εk in bands.eigenvalues]
    fp = if iszero(T)
        [zero.(εk) for εk in bands.eigenvalues]
    else
        [filled .* DFTK.Smearing.occupation_derivative.(model.smearing, (εk .- εF) ./ T) ./ T
         for εk in bands.eigenvalues]
    end
    (; f, fp)
end

"""
``(f_n - f_m)/(ε_n - ε_m)``, stable for ε_n ≈ ε_m, including the spin factor.
"""
function divided_difference(model, εn, εm, εF)
    DFTK.filled_occupation(model) *
        DFTK.Smearing.occupation_divided_difference(model.smearing, εn, εm, εF,
                                                    model.temperature)
end

"""
Group the band indices `idxs` (ascending) into degenerate manifolds, where consecutive
eigenvalues differ by less than `tol`.
"""
function degenerate_manifolds(ε, tol)
    manifolds = Vector{UnitRange{Int}}()
    start = 1
    for n = 2:length(ε)
        if ε[n] - ε[n-1] ≥ tol
            push!(manifolds, start:n-1)
            start = n
        end
    end
    push!(manifolds, start:length(ε))
    manifolds
end
