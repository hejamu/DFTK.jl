# Phase 5: Au(111)/water slab. NABEC tensors of the upper water molecule, compared with the
# same molecules in the identical cell without Au (vacuum reference, identical
# discretization, so the plane-wave gauge error largely cancels in the difference).
#
# Checks (handoff §4, Phase 5):
#  (a) Σ_atoms Z_xx,yy = (Ω/π) D̃_∥   [only with NABEC_ATOMS=all]
#  (b) D_zz ≈ 0; z-NABEC of water = FD z-dipole derivative   [NABEC_FDZ=1]
#  (c) per-molecule in-plane sum ≈ 0
#  (d) slab water vs vacuum water: the lateral image screening
#
# Env: NABEC_ECUT (20), NABEC_NK (4), NABEC_T (0.03), NABEC_ATOMS (water|all),
#      NABEC_FDZ (0|1), NABEC_REF (1|0)
using NABEC, DFTK, LinearAlgebra, Printf, Dates, Serialization
include(joinpath(@__DIR__, "..", "test", "slab_systems.jl"))

Ecut  = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
nk    = parse(Int, get(ENV, "NABEC_NK", "4"))
T     = parse(Float64, get(ENV, "NABEC_T", "0.03"))
which = get(ENV, "NABEC_ATOMS", "water")
do_fd = get(ENV, "NABEC_FDZ", "0") == "1"
do_ref = get(ENV, "NABEC_REF", "1") == "1"
tag   = "E$(Int(Ecut))_k$(nk)_T$(T)_$(which)"
mkpath("results")
logio = open("results/slab_$tag.txt", "w")
say(args...) = (println(args...); println(logio, args...); flush(logio); flush(stdout))
fmt(Z) = join([@sprintf("% .4f % .4f % .4f", Z[α, :]...) for α = 1:3], "\n        ")
stamp() = Dates.format(now(), "HH:MM:SS")

basis = au111_water(; Ecut, kgrid=[nk, nk, 1], temperature=T)
model = basis.model
ws    = upper_water_indices(basis)
atoms = which == "all" ? collect(1:length(model.positions)) : collect(ws)
say("[$(stamp())] Au(111)/H2O slab: $(length(model.positions)) atoms, Ecut=$Ecut, k=$(nk)x$(nk)x1, T=$T, NABEC atoms: $which")

scfres = self_consistent_field(basis; tol=1e-9, mixing=KerkerMixing(), callback=identity)
say("[$(stamp())] SCF done: n_iter=$(scfres.n_iter), εF=$(round(scfres.εF; digits=5))")

n_occ = maximum(count(>(1e-12), fk) for fk in scfres.occupation)
bands = NABEC.converged_bands(scfres.ham, n_occ + max(12, n_occ ÷ 4); tol=1e-9)
(; D, Dtilde) = drude_weight(basis, bands, scfres.εF)
say("[$(stamp())] (Ω/π)D  diag = ", round.(diag(D); digits=4))
say("           (Ω/π)D̃  diag = ", round.(diag(Dtilde); digits=4))

# Screened responses cost ≈1800 s each at 2×2 k: checkpoint them (keyed by setting), so a
# job that hits its time limit in the NABEC stage can restart from here.
ckpt = "results/slab_ph_$tag.jls"
ph = if isfile(ckpt)
    say("[$(stamp())] loading screened responses from $ckpt")
    (; atoms_saved, δVloc, δVbare, δρ) = deserialize(ckpt)
    @assert atoms_saved == atoms
    ScreenedPhonon(basis, atoms, δVloc, δVbare, δρ)
else
    p = ScreenedPhonon(scfres; atoms, tol=1e-8)
    serialize(ckpt, (; atoms_saved=atoms, p.δVloc, p.δVbare, p.δρ))
    p
end
say("[$(stamp())] screened responses done")
tol_stn = parse(Float64, get(ENV, "NABEC_TOL_STN", "1e-7"))
res = nabec_sternheimer(scfres; atoms, phonon=ph, bands, tol_sternheimer=tol_stn)
say("[$(stamp())] NABEC done")

Zw = [res.Z[findfirst(==(i), atoms)] for i in ws]
for (i, Z) in zip(ws, Zw)
    say("  Z[$(element_symbol(model.atoms[i]))#$i] = ", fmt(Z))
end
Zmol = sum(Zw)
say("  (c) molecule sum (slab)  = ", fmt(Zmol))
if which == "all"
    Zsum = sum(res.Z)
    say("  (a) Σ_all Z diag = ", round.(diag(Zsum); digits=4),
        "   vs (Ω/π)D̃ = ", round.(diag(Dtilde); digits=4),
        "   residual = ", round.(diag(Zsum) - diag(Dtilde); digits=4))
end
say("  (b) (Ω/π)D_zz = ", round(D[3, 3]; digits=5))

if do_fd
    h = 1e-3
    Zfd = map(ws) do i
        d = map((+1, -1)) do sgn
            positions = copy(model.positions)
            positions[i] = positions[i] + model.inv_lattice * [0, 0, sgn * h]
            b = PlaneWaveBasis(Model(model; positions); Ecut, kgrid=[nk, nk, 1],
                               fft_size=basis.fft_size)
            r = self_consistent_field(b; tol=1e-10, mixing=KerkerMixing(), callback=identity)
            dipole_moment(b, r.ρ)[3]
        end
        (d[1] - d[2]) / 2h
    end
    for (j, i) in enumerate(ws)
        say(@sprintf("  (b) Z_zz[%s#%d]: NABEC % .4f   FD z-dipole % .4f   diff % .4f",
                     element_symbol(model.atoms[i]), i, Zw[j][3, 3], Zfd[j], Zw[j][3, 3] - Zfd[j]))
    end
end

if do_ref
    bref  = au111_water(; Ecut, kgrid=[nk, nk, 1], with_gold=false)
    sref  = self_consistent_field(bref; tol=1e-10, callback=identity)
    wref  = 1:3    # upper molecule comes first when there is no gold
    rref  = nabec_sternheimer(sref; atoms=wref, tol_response=1e-10)
    say("[$(stamp())] vacuum reference (same cell, no Au):")
    for (j, i) in enumerate(wref)
        say("  Zvac[$(element_symbol(bref.model.atoms[i]))] = ", fmt(rref.Z[j]))
    end
    say("  (c) molecule sum (vac)   = ", fmt(sum(rref.Z)))
    say("  (d) Z_slab − Z_vac:")
    for j = 1:3
        say("      $(element_symbol(bref.model.atoms[j])): ", fmt(Zw[j] - rref.Z[j]))
    end
end
close(logio)
