# Split the H2O velocity-gauge ASR violation: evaluate Σ_κ Z^el with
#   (i)  Σ_κ H^τκβ (what nabec uses),
#   (ii) the ideal rigid translation -i[p_β, H],
#   (iii) (ii) with p replaced by the full velocity (reference: gives -N δ exactly only if
#        the basis is boost-invariant).
# Insulator: P = occupied bands, Z^el = -2 Σ_n f_n Im⟨δ⊥_v u_n|δ⊥_X u_n⟩.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

L    = parse(Float64, get(ENV, "NABEC_L", "12"))
Ecut = parse(Float64, get(ENV, "NABEC_ECUT", "20"))
xc     = get(ENV, "NABEC_XC", "lda")
basis  = water(; Ecut, L, functionals=(xc == "lda" ? LDA() : []))
println("xc = $xc, Ecut = $Ecut, L = $L")
model  = basis.model
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
ph     = ScreenedPhonon(scfres; tol=1e-11)
nocc   = count(>(0), scfres.occupation[1])
bands  = NABEC.converged_bands(scfres.ham, nocc)
ψO, ε  = bands.ψ[1], bands.eigenvalues[1]
Hk     = scfres.ham.blocks[1]
f      = 2.0
no_extra = zeros(eltype(ψO), size(ψO, 1), 0)
δ⊥(rhs) = DFTK.sternheimer_solver(Hk, ψO, ε, rhs; tol=1e-11, ψk_extra=no_extra,
                                  Hψk_extra=no_extra, εk_extra=zeros(0), maxiter=1000).δψk
v   = VelocityOperator(basis, 1)
δv  = [δ⊥(apply_velocity(v, α, ψO)) for α = 1:3]
Zel(X) = [-2f * sum(imag(dot(δv[α][:, j], X[β][:, j])) for j = 1:nocc) for α = 1:3, β = 1:3]

# (i) Σ_κ H^τ
Xτ = [δ⊥(sum(NABEC.apply_δH_cartesian(ph, 1, ψO, s)[β] for s = 1:length(model.positions)))
      for β = 1:3]
# (ii) -i[p̃_β, H] ψ_n = -i(ε_n p̃ψ_n - H p̃ψ_n)
Xp = map(1:3) do β
    pψ = apply_canonical(v, β, ψO)
    δ⊥(-im .* (pψ .* ε' .- Hk * pψ))
end
# (iii) same with the full velocity
Xv = map(1:3) do β
    vψ = apply_velocity(v, β, ψO)
    δ⊥(-im .* (vψ .* ε' .- Hk * vψ))
end
Nel = sum(charge_ionic, model.atoms)
for (label, X) in [("Σ_κ H^τ", Xτ), ("-i[p̃,H]", Xp), ("-i[v,H]", Xv)]
    Z = Zel(X) + Nel * I
    @printf "%-9s  ASR (Σ Z) diag = % .5f % .5f % .5f   max offdiag %.1e\n" label Z[1,1] Z[2,2] Z[3,3] maximum(abs, Z - Diagonal(Z))
end
