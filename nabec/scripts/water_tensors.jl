# Full H2O tensors: velocity-gauge NABEC (Sternheimer) and length-gauge APT (−∫ r δρ from the
# same screened response; equals the FD dipole APT to 5 digits). Writes results/water_tensors.tsv.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

L = parse(Float64, get(ENV, "NABEC_L", "12"))
mkpath("results")
out = open("results/water_tensors_L$(Int(L)).tsv", "w")
println(out, "L\tEcut\tatom\tmethod\talpha\tbeta\tZ")
for Ecut in parse.(Float64, split(get(ENV, "NABEC_ECUTS", "20,30,40"), ","))
    basis  = water(; Ecut, L)
    model  = basis.model
    scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
    ph     = ScreenedPhonon(scfres; tol=1e-11)
    vel    = nabec_sternheimer(scfres; phonon=ph).Z
    rs = r_vectors_cart(basis)
    c  = model.lattice * Vec3(0.5, 0.5, 0.5)
    len = map(1:3) do s
        Z = charge_ionic(model.atoms[s]) * Matrix(1.0I, 3, 3)
        for β = 1:3
            δρβ = sum(model.inv_lattice[α, β] .* ph.δρ[α, s][:, :, :, 1] for α = 1:3)
            Z[:, β] .+= -sum(δρβ[i] .* (rs[i] - c) for i in eachindex(rs)) * basis.dvol
        end
        Z
    end
    for s = 1:3, (m, Zs) in (("velocity", vel), ("length", len)), α = 1:3, β = 1:3
        @printf out "%.0f\t%.0f\t%d\t%s\t%d\t%d\t%.6f\n" L Ecut s m α β Zs[s][α, β]
    end
    flush(out)
    @printf "Ecut=%.0f done  max|vel−len| = %.4f\n" Ecut maximum(maximum(abs, vel[s] - len[s]) for s = 1:3)
end
close(out)
