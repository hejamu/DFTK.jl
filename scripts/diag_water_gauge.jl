# Locate the H2O velocity-gauge vs length-gauge discrepancy (Γ only, full diagonalization).
#  (a) v_nm vs i(ε_n-ε_m) r_nm for n occupied, all m, bucketed by ε_m
#  (b) Z from the same H^τ in three ways: velocity-gauge SOS, length-gauge SOS, -∫ r δρ
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

L    = parse(Float64, get(ENV, "NABEC_L", "12"))
Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
basis  = water(; Ecut, L)
model  = basis.model
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
ph     = ScreenedPhonon(scfres; tol=1e-11)
bands  = full_bands(scfres)
ψ, ε   = bands.ψ[1], bands.eigenvalues[1]
N      = length(ε)
O      = findall(>(0), scfres.occupation[1])
E      = setdiff(1:N, O)
f      = 2.0
kpt    = basis.kpoints[1]
c      = model.lattice * Vec3(0.5, 0.5, 0.5)
rs     = r_vectors_cart(basis)
println("n_G = $N, occupied = $(length(O))")

v = VelocityOperator(basis, 1)
Vmat = [ψ' * apply_velocity(v, α, ψ[:, O]) for α = 1:3]
# r|n⟩ projected on the basis (Galerkin)
Rmat = map(1:3) do α
    rψ = similar(ψ[:, O])
    for (j, n) in enumerate(O)
        ψr = ifft(basis, kpt, ψ[:, n])
        rψ[:, j] = fft(basis, kpt, [(r - c)[α] for r in rs] .* ψr)
    end
    ψ' * rψ
end

# (a) gauge identity, by energy bucket of m
Δ = ε .- ε[O]'          # Δ[m, j] = ε_m - ε_n
edges = [ε[maximum(O)], 1.0, 3.0, 10.0, Inf]
for α = 1:3
    lhs = Vmat[α][E, :]
    rhs = (-im .* Δ[E, :]) .* Rmat[α][E, :]     # v_nm... here ⟨m|v|n⟩ = i(ε_m-ε_n)⟨m|r|n⟩
    for b = 1:length(edges)-1
        sel = findall(m -> edges[b] < ε[m] ≤ edges[b+1], E)
        isempty(sel) && continue
        @printf "α=%d  ε_m ∈ (%.2f, %.2f]  n_m=%5d  ‖v - iΔr‖/‖v‖ = %.3e\n" α edges[b] edges[b+1] length(sel) norm(lhs[sel, :] - rhs[sel, :]) / norm(lhs[sel, :])
    end
end

# (b) Z three ways
for s = 1:length(model.positions)
    δHψ = NABEC.apply_δH_cartesian(ph, 1, ψ[:, O], s)
    Zvel = zeros(3, 3); Zlen = zeros(3, 3); Zrho = zeros(3, 3)
    for β = 1:3
        B = ψ' * δHψ[β]                        # B[m, j] = ⟨m|H^τ|n⟩
        for α = 1:3
            # velocity gauge: -2 f Σ Im[v_nm H_mn]/Δ²  (n occ, m empty)
            Zvel[α, β] = -2f * sum(imag.(conj.(Vmat[α][E, :]) .* B[E, :]) ./ Δ[E, :] .^ 2)
            # length gauge: -f Σ 2Re[r_nm H_mn/(ε_n - ε_m)]
            Zlen[α, β] = -2f * sum(real.(conj.(Rmat[α][E, :]) .* B[E, :]) ./ (-Δ[E, :]))
        end
        δρβ = sum(model.inv_lattice[α, β] .* ph.δρ[α, s][:, :, :, 1] for α = 1:3)
        Zrho[:, β] = -sum(δρβ[i] .* (rs[i] - c) for i in eachindex(rs)) * basis.dvol
    end
    println("atom $s  Z_el velocity | length | -∫rδρ (diagonal):")
    @printf "   vel %s\n   len %s\n   rho %s\n" round.(diag(Zvel); digits=5) round.(diag(Zlen); digits=5) round.(diag(Zrho); digits=5)
end
