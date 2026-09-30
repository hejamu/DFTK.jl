# Phase 5, step 1: fcc Au bulk (GTH-LDA semicore, 11 e⁻). D, D̃ and the sum rule vs k.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "slab_systems.jl"))

T    = parse(Float64, get(ENV, "NABEC_T", "0.03"))
Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
nks  = parse.(Int, split(get(ENV, "NABEC_NKS", "8,12,16"), ","))
mkpath("results")
out = open("results/au_bulk_T$(T).tsv", "w")
println(out, "nk\tT\tEcut\tZ\tD\tDt\tresid_D\tresid_Dt\tt_s")
for nk in nks
    t = @elapsed begin
        scfres = self_consistent_field(gold(; Ecut, kgrid=[nk, nk, nk], temperature=T);
                                       tol=1e-10, callback=identity)
        # Sternheimer route (= SOS to 1e-10): full_bands would store 16·N_G² bytes per
        # k-point, 15–30 GB on 16³–20³ meshes.
        n_occ = maximum(count(>(1e-12), fk) for fk in scfres.occupation)
        bands = NABEC.converged_bands(scfres.ham, n_occ + 12)
        (; D, Dtilde) = drude_weight(scfres, bands)
        res = nabec_sternheimer(scfres; bands)
    end
    Z = res.Z[1][1, 1]
    @printf "Au nk=%2d T=%.2f  Z = %.4f  ΩD/π = %.4f  ΩD̃/π = %.4f  Z−D = %+.4f  Z−D̃ = %+.4f  (%.0f s)\n" nk T Z D[1,1] Dtilde[1,1] Z-D[1,1] Z-Dtilde[1,1] t
    @printf out "%d\t%.3f\t%.1f\t%.6f\t%.6f\t%.6f\t%+.6f\t%+.6f\t%.0f\n" nk T Ecut Z D[1,1] Dtilde[1,1] Z-D[1,1] Z-Dtilde[1,1] t
    flush(out); flush(stdout)
end
close(out)
