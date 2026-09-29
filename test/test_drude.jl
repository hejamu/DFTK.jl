# Phase 2: Drude weight D and modified D̃.
using Test
using NABEC
using DFTK
using LinearAlgebra
using Printf

include("systems.jl")

@testset "Drude weight" begin
    @testset "empty lattice: ΩD/π = N_el" begin
        # No potential at all: free electrons folded into the fcc zone.
        lattice = 7.60 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
        model = Model(lattice; n_electrons=3, terms=[Kinetic()], symmetries=false,
                      temperature=0.04, smearing=Smearing.Gaussian(),
                      spin_polarization=:none)
        basis = PlaneWaveBasis(model; Ecut=3, kgrid=[20, 20, 20])  # BZ sampling error ~1e-3
        scfres = self_consistent_field(basis; tol=1e-10)
        (; D, Dtilde) = drude_weight(scfres, scf_bands(scfres))
        @printf "empty lattice  ΩD/π diag = %s, D̃ diag = %s\n" diag(D) diag(Dtilde)
        @test D ≈ Dtilde                          # no nonlocal term
        @test diag(D) ≈ fill(3.0, 3) rtol=1e-2
        @test norm(D - Diagonal(D)) < 1e-6        # cubic
    end
    @testset "fcc Al: D vs D̃" begin
        scfres = scf(aluminium(; kgrid=[8, 8, 8]))
        (; D, Dtilde) = drude_weight(scfres, scf_bands(scfres))
        @printf "Al 8³  ΩD/π = %.4f, ΩD̃/π = %.4f (per atom)\n" D[1,1] Dtilde[1,1]
        @test 1.5 < D[1, 1] < 2.5
        @test D[1, 1] ≈ D[2, 2] rtol=1e-3
    end
end
