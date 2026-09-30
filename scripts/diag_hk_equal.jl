# Is our fixed-G H_k(k') identical to DFTK's own Hamiltonian at k' when the G sets coincide?
# Ne atom, L = 8, Ecut 20. Compare full matrices (and their k-second differences).
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

L = 8.0; Ecut = 20.0
lattice = L * Matrix(1.0I, 3, 3)
Ne = ElementPsp(:Ne, GTH_LDA)
model  = model_DFT(lattice, [Ne], [lattice \ [L/2 + 0.1, L/2 + 0.05, L/2 - 0.07]];
                   functionals=LDA(), symmetries=false)
basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
G0 = DFTK.to_cpu(basis.kpoints[1].G_vectors)
function dftk_H(kc)
    b = PlaneWaveBasis(model; Ecut, kgrid=ExplicitKpoints([kc]), fft_size=basis.fft_size)
    Gk = DFTK.to_cpu(b.kpoints[1].G_vectors)
    Gk == G0 || return nothing          # G set changed: not comparable
    Matrix(Array(Hamiltonian(b; ρ=scfres.ρ).blocks[1]))
end
for h in (1e-3, 1e-2)
    for α = 1:3
        dk = model.recip_lattice \ Vec3(ntuple(i -> Float64(i == α), 3))
        Hd = [dftk_H(s * h * dk) for s in (-1, 0, 1)]
        any(isnothing, Hd) && (println("h=$h α=$α: G set changes, skipped"); continue)
        Ho = [NABEC.hamiltonian_at_shifted_k(scfres.ham, 1, s * h * dk) for s in (-1, 0, 1)]
        d0 = maximum(norm(Ho[i] - Hd[i]) / norm(Hd[i]) for i = 1:3)
        # second k-differences of the full matrices
        D2d = (Hd[3] - 2Hd[2] + Hd[1]) / h^2
        D2o = (Ho[3] - 2Ho[2] + Ho[1]) / h^2
        # decompose DFTK's second difference: kinetic part is exactly Diagonal(1) per α
        kinpart = norm(D2d - I) / norm(D2d)
        @printf "h=%.0e α=%s  ‖H_ours − H_DFTK‖/‖H‖ = %.2e   ‖∂²H ours − DFTK‖/‖∂²H‖ = %.2e   ‖∂²H_DFTK − 1‖/‖∂²H‖ = %.2e\n" h "xyz"[α] d0 norm(D2o - D2d) / norm(D2d) kinpart
    end
end
