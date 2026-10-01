# Phase 4: the Sternheimer split must reproduce the explicit sum over states.
using Test
using NABEC
using DFTK
using LinearAlgebra
using Printf

include("systems.jl")

@testset "Sternheimer = SOS" begin
    for (name, basis) in [("NaCl", nacl(; Ecut=12, kgrid=[2, 2, 2])),
                          ("Al",   aluminium(; kgrid=[4, 4, 4])),
                          ("Al4",  aluminium_conventional(; kgrid=[2, 2, 2]))]
        scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
        ph = ScreenedPhonon(scfres; tol=1e-10)
        sos = nabec_sos(scfres; phonon=ph)
        stn = nabec_sternheimer(scfres; phonon=ph)
        err = maximum(maximum(abs, a - b) for (a, b) in zip(sos.Z, stn.Z))
        @printf "%-5s  Z_1 SOS = %s\n       Z_1 Stn = %s\n       max diff %.2e  (n_P = %d)\n" name round.(diag(sos.Z[1]); digits=5) round.(diag(stn.Z[1]); digits=5) err stn.n_bands
        @test err < 1e-6
    end
end
