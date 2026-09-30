# Does an isolated Ne atom disperse with k in DFTK's own Hamiltonian (DFTK k-points,
# DFTK projectors, fixed SCF density)? Also compare with our shifted-k construction.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

L = 12.0; Ecut = 20.0
lattice = L * Matrix(1.0I, 3, 3)
for (name, el) in (("Ne", :Ne), ("H2", :H))
    if name == "Ne"
        atoms = [ElementPsp(:Ne, GTH_LDA)]; R = [[L/2 + 0.1, L/2 + 0.05, L/2 - 0.07]]
    else
        atoms = fill(ElementPsp(:H, GTH_LDA), 2); R = [[L/2 - 0.7, L/2 + 0.1, L/2 + 0.05], [L/2 + 0.7, L/2 + 0.1, L/2 + 0.05]]
    end
    model  = model_DFT(lattice, atoms, [lattice \ r for r in R]; functionals=LDA(), symmetries=false)
    basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
    scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
    nocc = count(>(0), scfres.occupation[1])
    println("$name: occupied ε at Γ = ", round.(scfres.eigenvalues[1][1:nocc]; digits=6))
    for kx in (0.0, 0.25, 0.5)
        kc = [kx, 0.0, 0.0]
        b  = PlaneWaveBasis(model; Ecut, kgrid=ExplicitKpoints([kc]), fft_size=basis.fft_size)
        ham = Hamiltonian(b; ρ=scfres.ρ)
        εd = eigvals(Hermitian(Matrix(Array(ham.blocks[1]))))[1:nocc]
        εo = eigvals(Hermitian(NABEC.hamiltonian_at_shifted_k(scfres.ham, 1, Vec3(kc))))[1:nocc]
        kcart = 2π / L * kx
        @printf "  k_x = %.2f (|k| = %.4f bohr⁻¹)  DFTK ε − ε(Γ): %s   fixed-G ε − ε(Γ): %s\n" kx kcart string(round.(εd .- scfres.eigenvalues[1][1:nocc]; sigdigits=3)) string(round.(εo .- scfres.eigenvalues[1][1:nocc]; sigdigits=3))
    end
end
