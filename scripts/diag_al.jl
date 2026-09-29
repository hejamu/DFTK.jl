# Diagnose the metal sum rule ΣZ = ΩD̃/π. For 1-atom Al, H^τ is a rigid translation and must
# equal -i[p, H]. Check it per k-point (bare and screened), and locate where it fails.
using NABEC, DFTK, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "test", "systems.jl"))

nk = parse(Int, get(ENV, "NABEC_NK", "6"))
T  = parse(Float64, get(ENV, "NABEC_T", "0.03"))
basis  = aluminium(; kgrid=[nk, nk, nk], temperature=T)
scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
ph     = ScreenedPhonon(scfres; tol=1e-11)
bands  = full_bands(scfres)
(; f) = NABEC.occupations_and_derivatives(basis, bands, scfres.εF)

# δρ of a rigid translation must be -∂_x ρ (reduced), δV_Hxc must be -∂_x V_Hxc.
ρ = scfres.ρ[:, :, :, 1]
ρG = fft(basis, ρ)
for α = 1:3
    dρ = real(ifft(basis, [-2π * im * G[α] for G in G_vectors(basis)] .* ρG))   # ∂ρ/∂x_α
    δρ = ph.δρ[α, 1][:, :, :, 1]
    @printf "α=%d  ‖δρ + ∂ρ‖/‖∂ρ‖ = %.3e   ∫δρ = %.2e\n" α norm(δρ + dρ)/norm(dρ) sum(δρ)*basis.dvol
end

worst = 0.0
for ik in 1:length(basis.kpoints)
    ψ, ε = bands.ψ[ik], bands.eigenvalues[ik]
    O  = findall(>(1e-12), f[ik])
    v  = VelocityOperator(basis, ik)
    δHψ = NABEC.apply_δH_cartesian(ph, ik, ψ[:, O], 1)
    for β = 1:3
        lhs = ψ' * δHψ[β]
        rhs = -im .* (ε[O]' .- ε) .* (ψ' * apply_canonical(v, β, ψ[:, O]))
        r = norm(lhs - rhs) / norm(rhs)
        global worst = max(worst, r)
        ik ≤ 3 && @printf "ik=%d β=%d  ‖H^τ + i[p,H]‖/‖·‖ = %.3e  (diag part: %.3e)\n" ik β r norm(diag(lhs[O, :])) / norm(rhs)
    end
end
@printf "worst translation-identity residual over k: %.3e\n" worst

res = nabec_sos(scfres; bands, phonon=ph)
(; D, Dtilde) = drude_weight(scfres, bands)
@printf "nk=%d T=%.3f  ΣZ = %.4f  ΩD/π = %.4f  ΩD̃/π = %.4f\n" nk T res.Z[1][1,1] D[1,1] Dtilde[1,1]
