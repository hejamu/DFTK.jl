# Term-by-term check of v = i[H, r] on H2O states (L = 12, Γ, Ecut 20), full diagonalization.
# For occupied n (localized) and empty m, compare
#   kinetic : ⟨m|p_α|n⟩            vs  i⟨m|(T r − r T)|n⟩
#   local   : 0                     vs  i⟨m|(V r − r V)|n⟩   (V = all local potentials)
#   nonlocal: ⟨m|∂_kα V_nl|n⟩       vs  i⟨m|(V_nl r − r V_nl)|n⟩
# r is applied in real space (box-centered) to the localized functions only.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

basis  = water(; Ecut=20.0, L=12.0)
model  = basis.model
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
bands  = full_bands(scfres)
ψ, ε   = bands.ψ[1], bands.eigenvalues[1]
O = findall(>(0), scfres.occupation[1]); E = setdiff(eachindex(ε), O)
kpt = basis.kpoints[1]; ham = scfres.ham; blk = ham.blocks[1]
c  = model.lattice * Vec3(0.5, 0.5, 0.5); rs = r_vectors_cart(basis)
rmul(α, X) = reduce(hcat, [fft(basis, kpt, [(r - c)[α] for r in rs] .* ifft(basis, kpt, X[:, j])) for j in axes(X, 2)])

# Split the Hamiltonian block into its operators
ops = blk.operators
kin = only(op for op in ops if op isa DFTK.FourierMultiplication)
nl  = only(op for op in ops if op isa DFTK.NonlocalOperator)
loc = [op for op in ops if op isa DFTK.RealSpaceMultiplication]
applyop(op, X) = op * X
Tψ(X)  = applyop(kin, X)
Vψ(X)  = sum(applyop(op, X) for op in loc)
Nψ(X)  = applyop(nl, X)
v = VelocityOperator(basis, 1)
ψO = ψ[:, O]
sets = [("lowest 8 empty", E[1:8]), ("empty 9–60", E[9:60]), ("all empty", E)]
for α = 1:3
    Pα   = ψ' * apply_canonical(v, α, ψO)
    Vnlα = ψ' * (apply_velocity(v, α, ψO) - apply_canonical(v, α, ψO))
    Kc   = im .* (ψ' * (Tψ(rmul(α, ψO)) - rmul(α, Tψ(ψO))))
    Lc   = im .* (ψ' * (Vψ(rmul(α, ψO)) - rmul(α, Vψ(ψO))))
    Nc   = im .* (ψ' * (Nψ(rmul(α, ψO)) - rmul(α, Nψ(ψO))))
    Rm   = ψ' * rmul(α, ψO)
    full = im .* (ε .- ε[O]') .* Rm                    # i(ε_m − ε_n) r_mn
    for (lab, S) in sets
        rel(a, b) = norm(a[S, :] - b[S, :]) / max(norm(b[S, :]), 1e-30)
        @printf "α=%s %-15s kinetic %.2e  local %.2e (‖local‖/‖v‖ %.2e)  nonlocal %.2e  total v vs iΔr %.2e\n" "xyz"[α] lab rel(Pα, Kc) rel(zero(Lc), Lc) norm(Lc[S, :]) / norm((Pα + Vnlα)[S, :]) rel(Vnlα, Nc) rel(Pα + Vnlα, full)
    end
end
