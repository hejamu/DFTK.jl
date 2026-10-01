# Decompose Z_zz(O) (and Z_xx(O)) into contributions from empty states m (by energy) and
# occupied orbitals n, for the velocity gauge (v_mn) and the length gauge (r_mn), same H^τ.
# Full diagonalization, L = 12, Γ.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
basis  = water(; Ecut, L=12.0)
model  = basis.model
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
ph     = ScreenedPhonon(scfres; atoms=[1], tol=1e-11)
bands  = full_bands(scfres)
ψ, ε   = bands.ψ[1], bands.eigenvalues[1]
O  = findall(>(0), scfres.occupation[1]); E = setdiff(eachindex(ε), O)
kpt = basis.kpoints[1]
c   = model.lattice * Vec3(0.5, 0.5, 0.5)
rs  = r_vectors_cart(basis)
v   = VelocityOperator(basis, 1)
δHψ = NABEC.apply_δH_cartesian(ph, 1, ψ[:, O], 1)
f   = 2.0
Δ   = ε[E] .- ε[O]'                  # ε_m − ε_n  (m empty rows, n occupied cols)
println("Ecut=$Ecut  n_G=$(length(ε))  ε_occ = ", round.(ε[O]; digits=4), "  LUMO = ", round(ε[E[1]]; digits=4))
for α in (3, 1)
    V  = (ψ' * apply_velocity(v, α, ψ[:, O]))[E, :]
    rψ = similar(ψ[:, O])
    for (j, n) in enumerate(O)
        rψ[:, j] = fft(basis, kpt, [(r - c)[α] for r in rs] .* ifft(basis, kpt, ψ[:, n]))
    end
    R = (ψ' * rψ)[E, :]
    B = (ψ' * δHψ[α])[E, :]           # Z_αα: displacement β = α
    cvel = -2f .* imag.(conj.(V) .* B) ./ Δ .^ 2          # per (m, n)
    clen = -2f .* real.(conj.(R) .* B) ./ (-Δ)
    @printf "\nZ^el_%s%s(O): velocity %.5f   length %.5f   diff %.5f\n" "xyz"[α] "xyz"[α] sum(cvel) sum(clen) sum(cvel) - sum(clen)
    println("  by occupied orbital n (ε_n):   vel      len      diff")
    for (j, n) in enumerate(O)
        @printf "    n=%d (%.4f)              % .5f % .5f % .5f\n" n ε[n] sum(cvel[:, j]) sum(clen[:, j]) sum(cvel[:, j]) - sum(clen[:, j])
    end
    println("  by empty-state energy window:  vel      len      diff    (identity residual ‖V − iΔR‖/‖V‖)")
    edges = [ε[E[1]] - 1e-9, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, Inf]
    for b = 1:length(edges)-1
        sel = findall(m -> edges[b] < ε[E[m]] ≤ edges[b+1], eachindex(E))
        isempty(sel) && continue
        id = norm(V[sel, :] - im .* Δ[sel, :] .* R[sel, :]) / norm(V[sel, :])
        @printf "    ε_m ∈ (%5.2f, %5.2f]  n=%5d  % .5f % .5f % .5f   %.3e\n" edges[b] edges[b+1] length(sel) sum(cvel[sel, :]) sum(clen[sel, :]) sum(cvel[sel, :]) - sum(clen[sel, :]) id
    end
end
