# Phase 5 cost probe: geometry sanity, one SCF, one screened phonon response, timings.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "slab_systems.jl"))

Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
nk   = parse(Int, get(ENV, "NABEC_NK", "2"))
basis = au111_water(; Ecut, kgrid=[nk, nk, 1])
model = basis.model
R = [model.lattice * x for x in model.positions]
println("atoms: ", length(R), "  n_electrons: ", model.n_electrons,
        "  n_G(k1): ", length(G_vectors(basis, basis.kpoints[1])),
        "  fft_size: ", basis.fft_size, "  n_k: ", length(basis.kpoints))
# minimum interatomic distance (minimum image, in-plane)
dmin = Inf
for i in eachindex(R), j in i+1:length(R), a in -1:1, b in -1:1
    d = norm(R[i] - R[j] + model.lattice * [a, b, 0])
    global dmin = min(dmin, d)
end
@printf "min distance %.3f bohr\n" dmin
ws = upper_water_indices(basis)
println("upper water atoms: ", ws, " ", element_symbol.(model.atoms[ws]))
for i in ws
    @printf "  %s  z = %.3f\n" element_symbol(model.atoms[i]) R[i][3]
end
t_scf = @elapsed scfres = self_consistent_field(basis; tol=1e-8,
                                                mixing=KerkerMixing())
@printf "SCF time: %.0f s, n_iter %d, εF = %.4f, n_bands %d\n" t_scf scfres.n_iter scfres.εF size(scfres.ψ[1], 2)
t_ph = @elapsed ScreenedPhonon(scfres; atoms=ws[1:1], tol=1e-8)
@printf "ScreenedPhonon (1 atom, 3 directions): %.0f s\n" t_ph
