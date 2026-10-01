# Is the velocity-gauge ASR violation a property of the discrete model?
# g_α(k) = Σ_n f_n ⟨u_n(k)|p̃_α|u_n(k)⟩ on the fixed Γ G-set; the velocity-gauge ASR Σ_κ Z_ακ,αα
# equals ∂g_α/∂k_α at Γ. Also the band curvature ∂²ε_n/∂k_α² of each occupied band (zero for
# an isolated molecule), from central differences of H_k(k ± h) eigenvalues.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

h = 1e-3
function probe_system(name, L; Ecut=parse(Float64, get(ENV, "NABEC_ECUT", "20")))
    name == "water" && return water(; Ecut, L)
    lattice = L * Matrix(1.0I, 3, 3)
    if name == "h2"      # GTH H: local only
        H = ElementPsp(:H, GTH_LDA)
        atoms, R = [H, H], [[L/2 - 0.7, L/2 + 0.1, L/2 + 0.05], [L/2 + 0.7, L/2 + 0.1, L/2 + 0.05]]
    elseif name == "ne"  # GTH Ne: closed shell, with nonlocal projectors
        Ne = ElementPsp(:Ne, GTH_LDA)
        atoms, R = [Ne], [[L/2 + 0.1, L/2 + 0.05, L/2 - 0.07]]
    end
    model = model_DFT(lattice, atoms, [lattice \ r for r in R]; functionals=LDA(), symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
end
sysname = get(ENV, "NABEC_SYS", "water")
for L in parse.(Float64, split(get(ENV, "NABEC_LS", "12,16"), ","))
    basis  = probe_system(sysname, L)
    model  = basis.model
    scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
    O  = findall(>(0), scfres.occupation[1]); f = 2.0
    k0 = basis.kpoints[1].coordinate
    kin = NABEC.kinetic_term(basis)
    Gc = [model.recip_lattice * G for G in DFTK.to_cpu(basis.kpoints[1].G_vectors)]
    @printf "%s Ecut = %s L = %.0f  (n_G = %d, nonlocal projectors: %d)\n" sysname get(ENV, "NABEC_ECUT", "20") L length(Gc) (isnothing(NABEC.nonlocal_term(basis)) ? 0 : size(NABEC.nonlocal_term(basis).ops[1].P, 2))
    for α = 1:3
        dk = model.recip_lattice \ Vec3(ntuple(i -> Float64(i == α), 3))
        res = map((-1, 0, 1)) do s
            kc = k0 + s * h * dk
            F  = eigen(Hermitian(NABEC.hamiltonian_at_shifted_k(scfres.ham, 1, kc)))
            kcart = model.recip_lattice * kc
            ψO = F.vectors[:, O]
            g  = f * sum(real(dot(ψO[:, j], ((p + kcart)[α] for p in Gc) .* ψO[:, j])) for j in eachindex(O))
            (; g, ε=F.values[O])
        end
        gp   = (res[3].g - res[1].g) / 2h
        curv = (res[3].ε .- 2res[2].ε .+ res[1].ε) ./ h^2
        @printf "  α=%s  ∂g/∂k = %.5f  (= velocity-gauge ASR_αα incl. −N_el δ: %.5f)   ∂²ε_n/∂k² = %s\n" "xyz"[α] gp gp - f * length(O) round.(curv; digits=4)
    end
end
