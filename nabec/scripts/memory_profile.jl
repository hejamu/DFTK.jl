# Peak memory (maxrss) per stage for isolated H2O, plus the sizes that drive it.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

gb() = Sys.maxrss() / 2^30
L    = parse(Float64, get(ENV, "NABEC_L", "12"))
Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
sos  = get(ENV, "NABEC_SOS", "1") == "1"
basis = water(; Ecut, L)
@printf "L=%.0f Ecut=%.0f  n_G=%d  fft=%s (%d pts)  after setup: %.2f GB\n" L Ecut length(G_vectors(basis, basis.kpoints[1])) basis.fft_size prod(basis.fft_size) gb()
scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
@printf "  SCF (n_bands %d):                  maxrss %.2f GB\n" size(scfres.ψ[1], 2) gb()
ph = ScreenedPhonon(scfres; tol=1e-10)
@printf "  ScreenedPhonon (9 responses):      maxrss %.2f GB   stored δV/δρ: %.3f GB\n" gb() (Base.summarysize(ph.δVloc) + Base.summarysize(ph.δVbare) + Base.summarysize(ph.δρ)) / 2^30
res = nabec_sternheimer(scfres; phonon=ph)
@printf "  NABEC Sternheimer (n_P %d):        maxrss %.2f GB\n" res.n_bands gb()
if sos
    b = full_bands(scfres)
    @printf "  full diagonalization (%d×%d):   maxrss %.2f GB\n" length(b.eigenvalues[1]) length(b.eigenvalues[1]) gb()
    nabec_sos(scfres; bands=b, phonon=ph)
    @printf "  NABEC SOS:                         maxrss %.2f GB\n" gb()
end
