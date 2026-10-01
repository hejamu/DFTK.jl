# Why is the slab Drude weight exactly zero? Cheap copy of the slab (low Ecut).
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "slab_systems.jl"))

basis  = au111_water(; Ecut=12, kgrid=[2, 2, 1], temperature=0.03)
scfres = self_consistent_field(basis; tol=1e-8, mixing=KerkerMixing(), callback=identity)
εF = scfres.εF
println("εF = $εF, T = $(basis.model.temperature), smearing = $(basis.model.smearing)")
n_occ = maximum(count(>(1e-12), fk) for fk in scfres.occupation)
println("n_occ (f > 1e-12) = $n_occ, SCF n_bands = $(size(scfres.ψ[1], 2))")
bands = NABEC.converged_bands(scfres.ham, n_occ + 12; tol=1e-9)
(; f, fp) = NABEC.occupations_and_derivatives(basis, bands, εF)
for ik in eachindex(basis.kpoints)
    ε = bands.eigenvalues[ik]
    near = findall(e -> abs(e - εF) < 0.1, ε)
    @printf "ik=%d  k=%s  bands within 0.1 Ha of εF: %d   max|f'| = %.3e   SCF-ε near εF: %d\n" ik basis.kpoints[ik].coordinate length(near) maximum(abs, fp[ik]) count(e -> abs(e - εF) < 0.1, scfres.eigenvalues[ik])
    ik == 1 && println("   ε - εF (bands): ", round.(ε[near] .- εF; digits=4))
    ik == 1 && println("   f' there      : ", round.(fp[ik][near]; sigdigits=3))
end
(; D, Dtilde) = drude_weight(basis, bands, εF)
println("D diag = ", diag(D), "   D̃ diag = ", diag(Dtilde))
(; D) = drude_weight(basis, NABEC.scf_bands(scfres), εF)
println("D from SCF bands = ", diag(D))
