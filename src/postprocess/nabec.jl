# Nonadiabatic Born effective charges (NABECs), Dreyer, Coh, Stengel, PRL 128, 095901 (2022),
# Eq. (7), at ω → 0 in the clean limit:
#
#   Z*_{α,κβ} = Z^ion_κ δ_αβ
#     − Im Σ_k w_k Σ_{n≠m} (f_n − f_m) / [(ε_n − ε_m)(ε_n − ε_m + iη)] v^α_nm H^{τκβ}_mn
#
# with v^α = ∂H_k/∂k_α (unscreened) and H^τ the screened first-order phonon Hamiltonian at
# q = 0 (bare local + nonlocal derivative + displaced core density in V_xc for NLCC
# + self-consistent δV_Hxc). For insulators this is
# the ordinary (Berry-phase) Born effective charge; for metals the intraband (Drude) term is
# excluded and Σ_κ Z*_κ = (Ω/π) D̃ (see `compute_drude_weight`).
#
# Evaluation splits the band space into P (explicitly computed, converged bands containing
# all partially occupied ones) and its complement Q (f = 0). For η → 0, pairing (n,m),(m,n):
#   Z^el = −Σ_k w_k [ Σ_{n≠m∈P} (f_n−f_m)/(ε_n−ε_m)² Im M_nm + 2 Σ_{n∈P} f_n Im⟨δ⊥v u_n|δ⊥τ u_n⟩ ]
# with M_nm = v_nm H^τ_mn and |δ⊥X u_n⟩ = −Q (H − ε_n)⁻¹ Q X|u_n⟩ from a Sternheimer solve.

"""
Screened first-order local potentials of the phonon perturbation at q = 0, for
reduced-coordinate displacements (α, atom): bare local derivative plus the self-consistent
δV_Hxc from [`solve_ΩplusK_split`](@ref). For metals the response includes the occupation
change and Fermi-level shift (the standard adiabatic static screening). Only potentials and
density responses are kept (not the orbital responses).
"""
function nabec_phonon_potentials(scfres; atoms=eachindex(scfres.basis.model.positions),
                                 tol=1e-8, verbose=false, mixing=nothing, kwargs...)
    basis = scfres.basis
    @assert length(basis.symmetries) == 1 "NABECs require symmetries=false"
    @assert basis.model.n_spin_components == 1 "Spin-polarized case not implemented"
    is_metal = !is_effective_insulator(basis, scfres.eigenvalues, scfres.εF)
    mixing = something(mixing, is_metal ? KerkerMixing() : SimpleMixing())
    atoms = collect(atoms)
    q0 = zero(Vec3{eltype(basis)})
    xc_terms = filter(t -> t isa TermXc, basis.terms)
    δVbare = map(Iterators.product(1:3, atoms)) do (α, s)
        δVloc = derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
            compute_local_potential(basis; positions=positions_αs)
        end
        # Nonlinear core correction: the core density moves rigidly with the atom
        for term in xc_terms
            δVcore = xc_core_displacement_potential(term, basis, α, s; ρ=scfres.ρ)
            isnothing(δVcore) || (δVloc = δVloc .+ δVcore[:, :, :, 1])
        end
        δVloc
    end
    δV = similar(δVbare)
    δρ = Matrix{Any}(undef, size(δVbare))
    for (is, s) in enumerate(atoms), α = 1:3
        δHextψ = compute_δHψ_αs(basis, scfres.ψ, α, s, q0; scfres.ρ)
        res = solve_ΩplusK_split(scfres, δHextψ; tol, verbose, mixing, kwargs...)
        δV[α, is] = δVbare[α, is] .+ res.δVind[:, :, :, 1]
        δρ[α, is] = res.δρ
    end
    (; atoms, δV, δVbare, δρ=identity.(δρ))
end

# Bare nonlocal ∂V_nl/∂x_α (reduced coordinates) of atom s applied to ψk at k-point ik.
function apply_nonlocal_displacement_derivative(basis, ik, α, s, ψk)
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    isempty(psp_groups) && return zero(ψk)
    kpt = basis.kpoints[ik]
    derivative_wrt_αs(model.positions, α, s) do positions_αs
        PDPψk(basis, positions_αs, psp_groups, kpt, kpt, ψk)
    end
end

# H^{τ_κβ} ψk for the three Cartesian displacements β of atom `potentials.atoms[iatom]`.
# ∂/∂R_β = Σ_α (L⁻¹)_{αβ} ∂/∂x_α with L the lattice and x reduced coordinates.
function apply_phonon_hamiltonian(basis, potentials, ik, ψk, iatom; screened=true)
    kpt = basis.kpoints[ik]
    red = map(1:3) do α
        V = screened ? potentials.δV[α, iatom] : potentials.δVbare[α, iatom]
        Vψ = similar(ψk)
        for n = 1:size(ψk, 2)
            Vψ[:, n] = fft(basis, kpt, V .* ifft(basis, kpt, ψk[:, n]))
        end
        Vψ .+ apply_nonlocal_displacement_derivative(basis, ik, α, potentials.atoms[iatom], ψk)
    end
    inv_lattice = basis.model.inv_lattice
    [sum(inv_lattice[α, β] .* red[α] for α = 1:3) for β = 1:3]
end

# Contribution of the ordered pairs (n, m) with n ∈ O or m ∈ O (O: occupied), within the
# band set ε: A = ψ' v ψ_O, B = ψ' H^τ ψ_O. Pairs closer than degeneracy_tol belong to the
# intraband (Drude) part and are skipped.
function nabec_pair_sum(model, ε, O, εF, A, B; η=0, degeneracy_tol)
    N = length(ε)
    notO = setdiff(1:N, O)
    function weight(n, m)
        Δ = ε[n] - ε[m]
        abs(Δ) < degeneracy_tol && return zero(complex(eltype(ε)))
        ratio = filled_occupation(model) *
                Smearing.occupation_divided_difference(model.smearing, ε[n], ε[m], εF,
                                                       model.temperature)
        ratio / (Δ + im * η)
    end
    W1 = [weight(n, m) for m = 1:N, n in O]
    W2 = [weight(n, m) for n in notO, m in O]
    sum(imag.(W1 .* conj.(A) .* B)) + sum(imag.(W2 .* A[notO, :] .* conj.(B[notO, :])))
end

@doc raw"""
    compute_nabec(scfres; atoms, n_bands, potentials, tol=1e-8, kwargs...)

Nonadiabatic Born effective charges of Dreyer, Coh, Stengel, PRL **128**, 095901 (2022)
at ``ω → 0`` (clean limit). Returns `(; Z, Z_electronic, Z_ionic, atoms, potentials)`, where
`Z[i]` is the ``3×3`` tensor of atom `atoms[i]` with `Z[i][α, β]` the current (dipole)
direction ``α`` and displacement direction ``β``, in units of the elementary charge.

For insulators these are the ordinary Born effective charges. For metals the intraband
(Drude) contribution is excluded, and ``\sum_κ Z_κ = (Ω/π) \tilde D`` holds once the
``k``-mesh resolves the smearing (see [`compute_drude_weight`](@ref)).

Keyword arguments:
- `atoms`: atoms for which to compute the tensors (default: all).
- `n_bands`: size of the explicitly treated band space (default: occupied bands plus a
  margin); its complement is handled by Sternheimer equations.
- `potentials`: precomputed result of `DFTK.nabec_phonon_potentials` (the screened phonon
  potentials, the expensive part), e.g. from a checkpoint.
- `tol`: tolerance of the screened response; `tol_sternheimer` (default `tol/10`) and
  `tol_bands` control the Sternheimer solves and the band diagonalization.

Requirements and caveats: `symmetries=false`; no spin polarization; no Hubbard, exact
exchange or magnetic terms. Nonlinear core corrections are supported: the displaced core
density enters the phonon perturbation (see `xc_core_displacement_potential`). Velocity-type quantities need pseudopotentials that converge in plane waves:
very hard projectors (e.g. GTH for O) converge extremely slowly with `Ecut`.
"""
function compute_nabec(scfres; atoms=eachindex(scfres.basis.model.positions),
                       n_bands=nothing, potentials=nothing, tol=1e-8,
                       tol_sternheimer=tol/10, tol_bands=tol/10, degeneracy_tol=1e-6,
                       occupation_threshold=1e-12, maxiter_sternheimer=500)
    basis = scfres.basis
    model = basis.model
    T = eltype(basis)
    @assert length(basis.symmetries) == 1 "NABECs require symmetries=false"
    @assert model.n_spin_components == 1 "Spin-polarized case not implemented"
    atoms = collect(atoms)
    potentials = something(potentials, nabec_phonon_potentials(scfres; atoms, tol))
    @assert potentials.atoms == atoms
    εF = scfres.εF

    n_occ = maximum(count(>(occupation_threshold), fk) for fk in scfres.occupation)
    n_bands = something(n_bands, n_occ + max(8, n_occ ÷ 4))
    diag = diagonalize_all_kblocks(lobpcg_hyper, scfres.ham, n_bands + 4;
                                   tol=tol_bands, miniter=1, maxiter=400)
    diag.converged || @warn "Band diagonalization for NABECs not converged"

    Zel = [zeros(T, 3, 3) for _ in atoms]
    for (ik, kpt) in enumerate(basis.kpoints)
        ψP = diag.X[ik][:, 1:n_bands]
        ε  = diag.λ[ik][1:n_bands]
        f  = filled_occupation(model) .*
             Smearing.occupation.(model.smearing, (ε .- εF) ./ model.temperature)
        O  = findall(>(occupation_threshold), f)
        maximum(O) < n_bands || error("n_bands too small: all bands of P are occupied")
        ψO = ψP[:, O]
        Hk = scfres.ham.blocks[ik]
        # Sternheimer in the complement of the whole of P (no Schur-complement extra bands)
        no_extra = similar(ψP, size(ψP, 1), 0)
        δ⊥(rhs) = sternheimer_solver(Hk, ψP, ε[O], rhs; tol=tol_sternheimer,
                                     ψk_extra=no_extra, Hψk_extra=no_extra,
                                     εk_extra=similar(ε, 0), maxiter=maxiter_sternheimer).δψk
        v   = velocity_operators(basis, ik)
        vψ  = [v[α](ψO) for α = 1:3]
        A   = [ψP' * vψ[α] for α = 1:3]
        δ⊥v = [δ⊥(vψ[α]) for α = 1:3]
        for ia in eachindex(atoms)
            Hτψ = apply_phonon_hamiltonian(basis, potentials, ik, ψO, ia)
            for β = 1:3
                B   = ψP' * Hτψ[β]
                δ⊥τ = δ⊥(Hτψ[β])
                for α = 1:3
                    SP = nabec_pair_sum(model, ε, O, εF, A[α], B; degeneracy_tol)
                    SQ = 2 * sum(f[O[j]] * imag(dot(δ⊥v[α][:, j], δ⊥τ[:, j]))
                                 for j in eachindex(O))
                    Zel[ia][α, β] -= basis.kweights[ik] * (SP + SQ)
                end
            end
        end
    end
    Z_electronic = [mpi_sum(Z, basis.comm_kpts) for Z in Zel]
    Z_ionic = [charge_ionic(model.atoms[s]) * Matrix{T}(I, 3, 3) for s in atoms]
    (; Z=Z_ionic .+ Z_electronic, Z_electronic, Z_ionic, atoms, potentials)
end

"""
Reference implementation of the NABECs by explicit sum over all states (full dense
diagonalization of every k-block). Only for small test systems. Supports a finite `η`.
"""
function compute_nabec_sos(scfres; atoms=eachindex(scfres.basis.model.positions),
                           potentials=nothing, tol=1e-8, η=0, degeneracy_tol=1e-6,
                           occupation_threshold=1e-12)
    basis = scfres.basis
    model = basis.model
    T = eltype(basis)
    atoms = collect(atoms)
    potentials = something(potentials, nabec_phonon_potentials(scfres; atoms, tol))
    εF = scfres.εF
    Zel = [zeros(T, 3, 3) for _ in atoms]
    for (ik, kpt) in enumerate(basis.kpoints)
        F  = eigen(Hermitian(Matrix(Array(scfres.ham.blocks[ik]))))
        ψk, ε = F.vectors, F.values
        f  = filled_occupation(model) .*
             Smearing.occupation.(model.smearing, (ε .- εF) ./ model.temperature)
        O  = findall(>(occupation_threshold), f)
        ψO = ψk[:, O]
        v  = velocity_operators(basis, ik)
        A  = [ψk' * v[α](ψO) for α = 1:3]
        for ia in eachindex(atoms)
            Hτψ = apply_phonon_hamiltonian(basis, potentials, ik, ψO, ia)
            for β = 1:3
                B = ψk' * Hτψ[β]
                for α = 1:3
                    Zel[ia][α, β] -= basis.kweights[ik] *
                                     nabec_pair_sum(model, ε, O, εF, A[α], B; η, degeneracy_tol)
                end
            end
        end
    end
    Z_electronic = [mpi_sum(Z, basis.comm_kpts) for Z in Zel]
    Z_ionic = [charge_ionic(model.atoms[s]) * Matrix{T}(I, 3, 3) for s in atoms]
    (; Z=Z_ionic .+ Z_electronic, Z_electronic, Z_ionic, atoms, potentials)
end
