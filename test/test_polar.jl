# Phase 3, test 2: rock-salt NaCl (LDA, GTH, no NLCC). ASR Σ_κ Z_κ = 0 and cubic symmetry.
# Literature LDA values Z*(Na) ≈ 1.1 (coarse settings here, so only a loose check).
using Test
using NABEC
using DFTK
using LinearAlgebra
using Printf

include("systems.jl")

@testset "NaCl: ASR and Z*" begin
    # ASR residual 0.20, 0.047, 0.013 at nk = 3, 4, 6 (velocity gauge needs BZ convergence)
    nk = parse(Int, get(ENV, "NABEC_NK", "6"))
    scfres = self_consistent_field(nacl(; Ecut=15, kgrid=[nk, nk, nk]); tol=1e-10,
                                   callback=identity)
    res = nabec_sternheimer(scfres)
    ZNa, ZCl = res.Z
    @printf "nk=%d  Z*(Na) = %.4f  Z*(Cl) = %.4f  ASR = %.2e  offdiag = %.1e\n" nk ZNa[1,1] ZCl[1,1] maximum(abs, ZNa + ZCl) maximum(abs, ZNa - Diagonal(ZNa))
    @test maximum(abs, ZNa + ZCl) < 2e-2
    @test maximum(abs, ZNa - ZNa[1, 1] * I) < 1e-6
    @test 0.9 < ZNa[1, 1] < 1.3
end
