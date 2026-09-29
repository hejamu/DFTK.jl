# Phase 3, test 3 (fcc Al): Z per atom vs k-mesh, D and D̃, sum rule Σ_κ Z = (Ω/π) D̃,
# for the 1-atom primitive and the 4-atom conventional cell. Writes results/al_sos.tsv.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

cells = get(ENV, "NABEC_CELLS", "prim,conv")
nks_prim = parse.(Int, split(get(ENV, "NABEC_NKS_PRIM", "4,6,8,10,12"), ","))
nks_conv = parse.(Int, split(get(ENV, "NABEC_NKS_CONV", "2,3,4,5,6"), ","))
temperature = parse(Float64, get(ENV, "NABEC_T", "0.01"))
Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "10"))
mkpath("results")
out = open("results/al_sos_T$(temperature).tsv", "w")
println(out, "cell\tnk\tT\tZxx_mean\tZyy_mean\tZzz_mean\tmax_offdiag\tsumZxx\tD_xx\tDt_xx\tresid_D\tresid_Dt\tt_s")

function run(cell, nk)
    basis = cell == "prim" ? aluminium(; Ecut, kgrid=[nk, nk, nk], temperature) :
                             aluminium_conventional(; Ecut, kgrid=[nk, nk, nk], temperature)
    t = @elapsed begin
        scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
        bands  = full_bands(scfres)
        (; D, Dtilde) = drude_weight(scfres, bands)
        res = nabec_sos(scfres; bands)
    end
    n_atoms = length(res.Z)
    Zmean = sum(res.Z) / n_atoms
    Zsum  = sum(res.Z)
    offd  = maximum(maximum(abs, Z - Diagonal(Z)) for Z in res.Z)
    @printf "%-4s nk=%2d  Z/atom = %.4f %.4f %.4f  ΣZ_xx = %.4f  ΩD/π = %.4f  ΩD̃/π = %.4f  ΣZ−D = %+.4f  ΣZ−D̃ = %+.4f  offdiag %.1e  (%.0f s)\n" cell nk Zmean[1,1] Zmean[2,2] Zmean[3,3] Zsum[1,1] D[1,1] Dtilde[1,1] Zsum[1,1]-D[1,1] Zsum[1,1]-Dtilde[1,1] offd t
    @printf out "%s\t%d\t%.4f\t%.6f\t%.6f\t%.6f\t%.2e\t%.6f\t%.6f\t%.6f\t%+.6f\t%+.6f\t%.0f\n" cell nk temperature Zmean[1,1] Zmean[2,2] Zmean[3,3] offd Zsum[1,1] D[1,1] Dtilde[1,1] Zsum[1,1]-D[1,1] Zsum[1,1]-Dtilde[1,1] t
    flush(out); flush(stdout)
end

occursin("prim", cells) && foreach(nk -> run("prim", nk), nks_prim)
occursin("conv", cells) && foreach(nk -> run("conv", nk), nks_conv)
close(out)
