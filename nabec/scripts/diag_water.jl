# Diagnose the H2O NABEC (SOS) vs FD-dipole mismatch.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))
include(joinpath(@__DIR__, "..", "test", "test_velocity.jl"))  # check_matrix_fd (runs its own tests too)

basis  = water(; Ecut=20, L=10)
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
model  = basis.model
n_atoms = length(model.positions)

# 1. Velocity operator at Γ (the only k-point here)
println("velocity FD check at Γ: ", check_matrix_fd(scfres; ik=1))

# 2. Length-gauge APT from DFTK's own δρ
ph = ScreenedPhonon(scfres; tol=1e-10)
rs = r_vectors_cart(basis)
c  = model.lattice * Vec3(0.5, 0.5, 0.5)
Zlen = map(1:n_atoms) do s
    Z = charge_ionic(model.atoms[s]) * Matrix(1.0I, 3, 3)
    for β = 1:3
        δρβ = sum(model.inv_lattice[α, β] .* ph.δρ[α, s][:, :, :, 1] for α = 1:3)
        Z[:, β] .+= -sum(δρβ[i] .* (rs[i] - c) for i in eachindex(rs)) * basis.dvol
    end
    Z
end
println("Length-gauge APT (−∫ r δρ):")
for s = 1:n_atoms; display(round.(Zlen[s]; digits=5)); end
println("ASR length gauge: ", round.(sum(Zlen); digits=5))

# 3. Translation identity  Σ_κ H^τκβ = -i[p_β, H]  on occupied states
bands = full_bands(scfres)
ψ, ε = bands.ψ[1], bands.eigenvalues[1]
nocc = count(>(0), scfres.occupation[1])
ψO = ψ[:, 1:nocc]
v  = VelocityOperator(basis, 1)
for β = 1:3
    ΣδH = sum(NABEC.apply_δH_cartesian(ph, 1, ψO, s)[β] for s = 1:n_atoms)
    lhs = ψ' * ΣδH
    P   = ψ' * apply_canonical(v, β, ψO)
    rhs = -im .* (ε[1:nocc]' .- ε) .* P
    @printf "β=%d  ‖ΣH^τ - (-i[p,H])‖/‖.‖ = %.3e   (bare: %.3e)\n" β norm(lhs - rhs)/norm(rhs) begin
        ΣδH0 = sum(NABEC.apply_δH_cartesian(ph, 1, ψO, s; screened=false)[β] for s = 1:n_atoms)
        lhs0 = ψ' * ΣδH0
        # bare translation: -i[p, V_ext] only (no Hxc)
        norm(lhs0 - rhs) / norm(rhs)
    end
end

# 4. SOS NABEC, screened and bare
res = nabec_sos(scfres; bands, phonon=ph)
println("SOS NABEC:")
for s = 1:n_atoms; display(round.(res.Z[s]; digits=5)); end
println("ASR SOS: ", round.(sum(res.Z); digits=5))
