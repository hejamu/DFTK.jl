# Chase the H2O velocity- vs length-gauge gap (L = 12, Γ). Variants:
#   lda    : LDA + full GTH (the production Hamiltonian)
#   nonl   : LDA, no nonlocal pseudopotential term (then v = p exactly)
#   x_raw  : exchange-only LDA via LocalNonlinearity, f ∝ ρ^{4/3} (non-analytic as ρ → 0)
#   x_reg  : same, regularized f ∝ (ρ+ρ0)^{4/3} − ρ0^{4/3}, smooth in the vacuum tails
# Output: max |Z_vel − Z_len| over all components, ASR of both, HOMO–LUMO gap.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

const CX = 3/4 * (3/π)^(1/3)
slater_raw(ρ) = -CX * max(ρ, 1e-14)^(4/3)
const ρ0 = parse(Float64, get(ENV, "NABEC_RHO0", "1e-3"))
slater_reg(ρ) = -CX * ((max(ρ, -ρ0/2) + ρ0)^(4/3) - ρ0^(4/3))

const DOJO = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")

function water_variant(variant; Ecut, L=parse(Float64, get(ENV, "NABEC_L", "12")))
    b0 = water(; Ecut, L)                 # geometry and atoms from the standard builder
    m0 = b0.model
    if variant in ("dojo", "dojo_gth_h")
        # PseudoDojo NC (ONCVPSP, designed for plane waves); NLCC off because DFTK's phonon
        # δH does not include it, so the translation identity stays exact.
        Oel = ElementPsp(:O, DOJO)
        Hel = variant == "dojo" ? ElementPsp(:H, DOJO) : m0.atoms[2]
        atoms = [Oel, Hel, Hel]
        terms = [Kinetic(), AtomicLocal(), AtomicNonlocal(), Ewald(), PspCorrection(), Hartree(),
                 Xc([:lda_x, :lda_c_pw]; use_nlcc=false)]
        model = Model(m0.lattice, atoms, m0.positions; terms, symmetries=false,
                      spin_polarization=:none)
        return PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
    end
    base = [Kinetic(), AtomicLocal(), Ewald(), PspCorrection(), Hartree()]
    terms = variant == "lda"   ? [base..., AtomicNonlocal(), LDA()] :
            variant == "nonl"  ? [base..., LDA()] :
            variant == "x_raw" ? [base..., AtomicNonlocal(), LocalNonlinearity(slater_raw)] :
            variant == "x_reg" ? [base..., AtomicNonlocal(), LocalNonlinearity(slater_reg)] :
            error("unknown variant $variant")
    model = Model(m0.lattice, m0.atoms, m0.positions; terms, symmetries=false,
                  spin_polarization=:none)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
end

function length_gauge(basis, ph)
    model = basis.model
    rs = r_vectors_cart(basis)
    c  = model.lattice * Vec3(0.5, 0.5, 0.5)
    map(1:length(model.positions)) do s
        Z = charge_ionic(model.atoms[s]) * Matrix(1.0I, 3, 3)
        for β = 1:3
            δρβ = sum(model.inv_lattice[α, β] .* ph.δρ[α, s][:, :, :, 1] for α = 1:3)
            Z[:, β] .+= -sum(δρβ[i] .* (rs[i] - c) for i in eachindex(rs)) * basis.dvol
        end
        Z
    end
end

mkpath("results")
out = open("results/water_variants_$(get(ENV, "NABEC_TAG", "run")).tsv", "w")
println(out, "variant\tEcut\tmax_dev\tasr_vel\tasr_len\tgap\tdOxx\tdOyy\tdOzz\tdHxx\tdHyy\tdHzz")
for spec in split(get(ENV, "NABEC_RUNS", "lda:50"), ",")
    variant, E = split(spec, ":")
    Ecut = parse(Float64, E)
    basis  = water_variant(variant; Ecut)
    scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
    n_occ  = count(>(0), scfres.occupation[1])
    bands  = NABEC.converged_bands(scfres.ham, n_occ + 4)
    gap    = bands.eigenvalues[1][n_occ+1] - bands.eigenvalues[1][n_occ]
    ph     = ScreenedPhonon(scfres; tol=1e-11)
    vel    = nabec_sternheimer(scfres; phonon=ph).Z
    len    = length_gauge(basis, ph)
    dev    = maximum(maximum(abs, vel[s] - len[s]) for s = 1:3)
    dO, dH = diag(vel[1] - len[1]), diag(vel[2] - len[2])
    @printf "%-6s L=%s Ecut=%3.0f  max|vel−len| = %.4f   ASR vel %.4f  len %.4f   gap %.3f Ha   ΔO %s  ΔH %s\n" variant get(ENV, "NABEC_L", "12") Ecut dev maximum(abs, sum(vel)) maximum(abs, sum(len)) gap round.(dO; digits=4) round.(dH; digits=4)
    @printf out "%s\t%.0f\t%.6f\t%.6f\t%.6f\t%.4f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" variant Ecut dev maximum(abs, sum(vel)) maximum(abs, sum(len)) gap dO... dH...
    flush(out); flush(stdout)
end
close(out)
