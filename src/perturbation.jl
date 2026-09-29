# Screened first-order phonon Hamiltonian at q = 0,
#   H^{τ_κβ} = ∂V_loc/∂τ + ∂V_nl/∂τ + δV_Hxc[δρ],
# for a Cartesian displacement of atom κ along β.
#
# The bare part comes from DFTK's phonon machinery (reduced coordinates); the
# self-consistent δV_Hxc from `solve_ΩplusK_split`, which at q = 0 in metals includes the
# occupation change and Fermi-level shift δε_F (the standard adiabatic prescription, as in
# Dreyer et al.). All local pieces are stored as one real-space potential per displacement,
# so H^τ can be applied to any set of orbitals, one k-point at a time.

"""
Self-consistent responses to reduced-coordinate displacements (α_red, atom).
`δVloc[α, i]` is ∂V_loc/∂x_α plus (if screened) δV_Hxc, as a real-space array.
"""
struct ScreenedPhonon{TV, Tρ}
    basis::PlaneWaveBasis
    atoms::Vector{Int}
    δVloc::Matrix{TV}      # bare local + induced Hxc potential, [α_red, iatom]
    δVbare::Matrix{TV}     # bare local only, [α_red, iatom]
    δρ::Matrix{Tρ}
end

function ScreenedPhonon(scfres; atoms=1:length(scfres.basis.model.positions),
                        tol=1e-10, verbose=false,
                        mixing=is_metal(scfres) ? KerkerMixing() : SimpleMixing(), kwargs...)
    basis = scfres.basis
    @assert length(basis.symmetries) == 1 "Use symmetries=false"
    atoms = collect(atoms)
    q0 = zero(Vec3{Float64})
    δVbare = [DFTK.derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
                  DFTK.compute_local_potential(basis; positions=positions_αs)
              end for α = 1:3, s in atoms]
    res = [begin
               δHextψ = DFTK.compute_δHψ_αs(basis, scfres.ψ, α, s, q0)
               DFTK.solve_ΩplusK_split(scfres, δHextψ; tol, verbose, mixing, kwargs...)
           end for α = 1:3, s in atoms]
    # δVind is fft_size × n_spin; only the spin-unpolarized case is supported
    δVloc = map((Vb, r) -> Vb .+ r.δVind[:, :, :, 1], δVbare, res)
    ScreenedPhonon(basis, atoms, δVloc, δVbare, map(r -> r.δρ, res))
end

is_metal(scfres) = !DFTK.is_effective_insulator(scfres.basis, scfres.eigenvalues, scfres.εF)

"""
Multiply the columns of ψk (Fourier coefficients at k-point `ik`) by the real-space
potential V.
"""
function apply_local(basis, ik, V, ψk)
    kpt = basis.kpoints[ik]
    out = similar(ψk)
    for n = 1:size(ψk, 2)
        ψr = ifft(basis, kpt, ψk[:, n])
        out[:, n] = fft(basis, kpt, V .* ψr)
    end
    out
end

"""
Bare nonlocal ∂V_nl/∂x_α (reduced) for atom s applied to ψk at k-point `ik`.
"""
function apply_nonlocal_derivative(basis, ik, α, s, ψk)
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    isempty(psp_groups) && return zero(ψk)
    kpt = basis.kpoints[ik]
    DFTK.derivative_wrt_αs(model.positions, α, s) do positions_αs
        DFTK.PDPψk(basis, positions_αs, psp_groups, kpt, kpt, ψk)
    end
end

"""
``H^{τ}ψ_k`` for the reduced displacement (α_red, iatom) at k-point `ik`.
"""
function apply_δH_reduced(ph::ScreenedPhonon, ik, ψk, α, iatom; screened=true)
    V = screened ? ph.δVloc[α, iatom] : ph.δVbare[α, iatom]
    apply_local(ph.basis, ik, V, ψk) .+
        apply_nonlocal_derivative(ph.basis, ik, α, ph.atoms[iatom], ψk)
end

"""
``H^{τ_κβ}ψ_k`` for Cartesian displacements β = 1:3 of atom `ph.atoms[iatom]`.
∂/∂R_β = Σ_α (L⁻¹)_{αβ} ∂/∂x_α, with L the lattice matrix and x reduced coordinates.
"""
function apply_δH_cartesian(ph::ScreenedPhonon, ik, ψk, iatom; screened=true)
    inv_lattice = ph.basis.model.inv_lattice
    red = [apply_δH_reduced(ph, ik, ψk, α, iatom; screened) for α = 1:3]
    [sum(inv_lattice[α, β] .* red[α] for α = 1:3) for β = 1:3]
end
