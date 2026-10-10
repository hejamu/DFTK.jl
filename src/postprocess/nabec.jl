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
# The Q term is evaluated as 2 Σ f_n Im⟨w_n|H^τ u_n⟩ with w_n = δ⊥(δ⊥v u_n), so the number of
# Sternheimer solves does not grow with the number of atoms.

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

# Column range of atom s in the projector matrix of the AtomicNonlocal term (pseudopotential
# groups in model order, atoms within a group, count_n_proj(psp) columns each).
function nonlocal_projector_columns(model, s)
    offset = 0
    for group in model.atom_groups
        element = model.atoms[first(group)]
        element isa ElementPsp || continue
        n = count_n_proj(element.psp)
        for i in group
            i == s && return offset .+ (1:n)
            offset += n
        end
    end
    nothing
end

# Bare nonlocal ∂V_nl/∂x_α (reduced coordinates) of atom s applied to ψk at k-point ik.
# Only atom s's projectors depend on x_s, through the structure factor e^{-2πi(k+G)·x_s}, so
# ∂P_s = -2πi (k+G)_α P_s and ∂(P D P') = ∂P_s D_s P_s' + P_s D_s ∂P_s'. Cost O(N_PW n_proj(s)).
function apply_nonlocal_displacement_derivative(basis, ik, α, s, ψk)
    iterm = findfirst(t -> t isa TermAtomicNonlocal, basis.terms)
    isnothing(iterm) && return zero(ψk)
    cols = nonlocal_projector_columns(basis.model, s)
    isnothing(cols) && return zero(ψk)
    op = basis.terms[iterm].ops[ik]
    Ps = op.P[:, cols]
    Ds = op.D[cols, cols]
    T  = eltype(basis)
    ∂Ps = map(p -> -2T(π) * im * p[α], Gplusk_vectors(basis, basis.kpoints[ik])) .* Ps
    ∂Ps * (Ds * (Ps' * ψk)) .+ Ps * (Ds * (∂Ps' * ψk))
end

# Reference implementation by automatic differentiation through all projectors (tests only).
function apply_nonlocal_displacement_derivative_ad(basis, ik, α, s, ψk)
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    isempty(psp_groups) && return zero(ψk)
    kpt = basis.kpoints[ik]
    derivative_wrt_αs(model.positions, α, s) do positions_αs
        PDPψk(basis, positions_αs, psp_groups, kpt, kpt, ψk)
    end
end

# Cartesian displacement derivatives β of atom s: ∂/∂R_β = Σ_α (L⁻¹)_{αβ} ∂/∂x_α.
function apply_nonlocal_displacement_derivative_cart(basis, ik, s, ψk)
    red = [apply_nonlocal_displacement_derivative(basis, ik, α, s, ψk) for α = 1:3]
    inv_lattice = basis.model.inv_lattice
    [sum(inv_lattice[α, β] .* red[α] for α = 1:3) for β = 1:3]
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
        # ⟨δ⊥v u|δ⊥τ u⟩ = ⟨w|H^τ u⟩ with w = Q(H-ε)⁻¹Q(H-ε)⁻¹Q v u = δ⊥(δ⊥v), since the projected
        # resolvent is Hermitian: two Sternheimer solves per band for the velocity replace one
        # per band and phonon perturbation.
        w   = [δ⊥(δ⊥(vψ[α])) for α = 1:3]
        for ia in eachindex(atoms)
            Hτψ = apply_phonon_hamiltonian(basis, potentials, ik, ψO, ia)
            for β = 1:3
                B = ψP' * Hτψ[β]
                for α = 1:3
                    SP = nabec_pair_sum(model, ε, O, εF, A[α], B; degeneracy_tol)
                    SQ = 2 * sum(f[O[j]] * imag(dot(w[α][:, j], Hτψ[β][:, j]))
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

"""
Bare (unscreened) first-order phonon potentials at q = 0: the local-potential derivative plus,
for models with nonlinear core correction, the displaced-core XC term. Same layout as
[`nabec_phonon_potentials`](@ref) (with `δV = δVbare`), but without any response solve.
"""
function nabec_bare_potentials(scfres; atoms=eachindex(scfres.basis.model.positions))
    basis = scfres.basis
    model = basis.model
    T = eltype(basis)
    atoms = collect(atoms)
    xc_terms = filter(t -> t isa TermXc, basis.terms)
    # Local pseudopotential: V(G) = Σ_s f_s(|G|) e^{-2πi G·x_s} / √Ω, so the displacement of
    # atom s only changes its own term: ∂V/∂x_sα (G) = -2πi G_α f_s(|G|) e^{-2πi G·x_s} / √Ω.
    form_factors, iG2ifnorm = atomic_local_form_factors(basis)
    group_of = Dict(i => ig for (ig, group) in enumerate(model.atom_groups) for i in group)
    Gs = vec(G_vectors(basis))
    δVbare = map(Iterators.product(1:3, atoms)) do (α, s)
        ff = @view form_factors[:, group_of[s]]
        x  = model.positions[s]
        δV_fourier = map(eachindex(Gs)) do iG
            -2T(π) * im * Gs[iG][α] * cis2pi(-dot(Gs[iG], x)) * ff[iG2ifnorm[iG]] /
                sqrt(model.unit_cell_volume)
        end
        δVloc = irfft(basis, reshape(δV_fourier, basis.fft_size))
        for term in xc_terms
            δVcore = xc_core_displacement_potential(term, basis, α, s; ρ=scfres.ρ)
            isnothing(δVcore) || (δVloc = δVloc .+ δVcore[:, :, :, 1])
        end
        δVloc
    end
    (; atoms, δV=δVbare, δVbare, δρ=nothing)
end

# Pair matrix M of the explicit band space P: Σ_{n≠m} F_nm v_nm X_mn = Σ_mn M[m, n] X[m, n]
# with F_nm = (f_n - f_m) / (ε_n - ε_m)² (Drude pairs excluded), restricted to pairs with at
# least one occupied band, as in `nabec_pair_sum`. A = ψP' v ψ_O.
function nabec_pair_matrix(model, ε, O, εF, A; degeneracy_tol)
    N = length(ε)
    M = zeros(complex(eltype(ε)), N, N)
    function weight(n, m)
        Δ = ε[n] - ε[m]
        abs(Δ) < degeneracy_tol && return zero(eltype(ε))
        filled_occupation(model) *
            Smearing.occupation_divided_difference(model.smearing, ε[n], ε[m], εF,
                                                   model.temperature) / Δ
    end
    for (jn, n) in enumerate(O), m = 1:N
        M[m, n] = weight(n, m) * conj(A[m, jn])          # v_nm = conj(⟨m|v|n⟩)
    end
    for n in setdiff(1:N, O), (jm, m) in enumerate(O)
        M[m, n] = weight(n, m) * A[n, jm]                # v_nm = ⟨n|v|m⟩
    end
    M
end

@doc raw"""
    compute_nabec_field(scfres; atoms, n_bands, tol=1e-8, kwargs...)

Same quantity as [`compute_nabec`](@ref), evaluated with the self-consistent response to the
velocity ("field") perturbation instead of one screened phonon response per atom and
direction (interchange theorem). Splitting ``H^τ = H^τ_{\rm b} + K δρ_τ`` with
``δρ_τ = (1-χ_0K)^{-1}χ_0 H^τ_{\rm b}`` and using the symmetry of ``χ_0`` and ``K``,
```math
Z^{\rm el}_{κ,αβ} = B(v^α, H^{τ_{κβ}}_{\rm b}) + \operatorname{Tr}\big[γ^{(1)}[δV^α]\, H^{τ_{κβ}}_{\rm b}\big],
\qquad δV^α = K\,(1-χ_0K)^{-1}\,\tilde ρ^α ,
```
where ``B`` is the band sum of Eq. (7) with the bare phonon Hamiltonian,
``\tilde ρ^α(r) = -\operatorname{Im}\sum_{n≠m}F_{nm}v^α_{nm}ψ_m^*(r)ψ_n(r)`` and ``γ^{(1)}[δV]``
the (static, non-self-consistent) first-order density matrix of the potential ``δV``. The
cost is three screened responses plus band-wise Sternheimer solves for the velocity,
independent of the number of atoms; each atom only needs its bare ``H^τ``.
"""
function compute_nabec_field(scfres; atoms=eachindex(scfres.basis.model.positions),
                             n_bands=nothing, tol=1e-8, tol_sternheimer=tol/10,
                             tol_bands=tol/10, degeneracy_tol=1e-6,
                             occupation_threshold=1e-12, maxiter_sternheimer=500,
                             mixing=nothing, verbose=false)
    basis = scfres.basis
    model = basis.model
    T = eltype(basis)
    @assert length(basis.symmetries) == 1 "NABECs require symmetries=false"
    @assert model.n_spin_components == 1 "Spin-polarized case not implemented"
    atoms = collect(atoms)
    εF = scfres.εF
    bare = @timing "nabec: bare potentials" nabec_bare_potentials(scfres; atoms)

    n_occ = maximum(count(>(occupation_threshold), fk) for fk in scfres.occupation)
    n_bands = something(n_bands, n_occ + max(8, n_occ ÷ 4))
    diag = diagonalize_all_kblocks(lobpcg_hyper, scfres.ham, n_bands + 4;
                                   tol=tol_bands, miniter=1, maxiter=400)
    diag.converged || @warn "Band diagonalization for NABECs not converged"

    # 1. Band sum with the bare phonon Hamiltonian, and the source densities ρ̃^α
    verbose && @info "compute_nabec_field: bands done" n_bands
    Zel = [zeros(T, 3, 3) for _ in atoms]
    ψ_src  = [[similar(diag.X[ik], 0, 0) for ik in eachindex(basis.kpoints)] for _ = 1:3]
    δψ_src = deepcopy(ψ_src)
    occ_src = [[zeros(T, 0) for _ in eachindex(basis.kpoints)] for _ = 1:3]
    for (ik, kpt) in enumerate(basis.kpoints)
        ψP = diag.X[ik][:, 1:n_bands]
        ε  = diag.λ[ik][1:n_bands]
        f  = filled_occupation(model) .*
             Smearing.occupation.(model.smearing, (ε .- εF) ./ model.temperature)
        O  = findall(>(occupation_threshold), f)
        maximum(O) < n_bands || error("n_bands too small: all bands of P are occupied")
        ψO = ψP[:, O]
        Hk = scfres.ham.blocks[ik]
        no_extra = similar(ψP, size(ψP, 1), 0)
        δ⊥(rhs) = sternheimer_solver(Hk, ψP, ε[O], rhs; tol=tol_sternheimer,
                                     ψk_extra=no_extra, Hψk_extra=no_extra,
                                     εk_extra=similar(ε, 0), maxiter=maxiter_sternheimer).δψk
        v  = velocity_operators(basis, ik)
        vψ = [v[α](ψO) for α = 1:3]
        A  = [ψP' * vψ[α] for α = 1:3]
        w  = @timing "nabec: velocity Sternheimer" [δ⊥(δ⊥(vψ[α])) for α = 1:3]  # Q(H-ε)⁻²Q v u_n
        # Nonlocal part of B; its local part is ∫ ρ̃ δV_loc (below), with no band loop
        @timing "nabec: bare band sum" for ia in eachindex(atoms)
            Hτψ = apply_nonlocal_displacement_derivative_cart(basis, ik, atoms[ia], ψO)
            for β = 1:3
                B = ψP' * Hτψ[β]
                for α = 1:3
                    SP = nabec_pair_sum(model, ε, O, εF, A[α], B; degeneracy_tol)
                    SQ = 2 * sum(f[O[j]] * imag(dot(w[α][:, j], Hτψ[β][:, j]))
                                 for j in eachindex(O))
                    Zel[ia][α, β] -= basis.kweights[ik] * (SP + SQ)
                end
            end
        end
        # ρ̃^α = -Im Σ_mn M_mn ψ_m^* ψ_n (P pairs) + 2 Σ_n f_n Im ψ_n^* w_n (Q part), written in
        # the form Σ_j occ_j 2Re ψ_j^* δψ_j of `compute_δρ`.
        for α = 1:3
            M = nabec_pair_matrix(model, ε, O, εF, A[α]; degeneracy_tol)
            ψ_src[α][ik]   = hcat(ψP, ψO)
            δψ_src[α][ik]  = hcat((im / 2) .* (ψP * transpose(M)), -im .* w[α])
            occ_src[α][ik] = vcat(ones(T, n_bands), f[O])
        end
    end
    ρ̃ = [compute_δρ(basis, ψ_src[α], δψ_src[α], occ_src[α]) for α = 1:3]
    verbose && @info "compute_nabec_field: band sum and source densities done"
    ψ_src = δψ_src = nothing

    # 2. Screened field responses δV^α = K (1 - χ0 K)⁻¹ ρ̃^α and their density matrices γ⁽¹⁾
    is_metal = !is_effective_insulator(basis, scfres.eigenvalues, εF)
    mixing = something(mixing, is_metal ? KerkerMixing() : SimpleMixing())
    bandtolalg = BandtolBalanced(scfres)
    ε_adj = DielectricAdjoint(scfres; bandtolalg)
    ρ = scfres.ρ
    precon = FunctionPreconditioner() do Pδρ, δρ
        Pδρ .= vec(mix_density(mixing, basis, reshape(δρ, size(ρ)); ham=scfres.ham, basis,
                               ρin=ρ, εF, scfres.eigenvalues, scfres.ψ))
    end
    responses = @timing "nabec: field responses" map(1:3) do α
        info = inexact_gmres(ε_adj, vec(ρ̃[α]); tol, precon, krylovdim=20, maxiter=100, s=100,
                             callback=identity)
        info.converged || @warn "Field response not converged" α
        δV = apply_kernel(basis, reshape(info.x, size(ρ)); ρ)
        verbose && @info "field response α=$α: $(info.n_iter) GMRES iterations, converged=$(info.converged)"
        apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, εF, scfres.eigenvalues,
                    multiply_ψ_by_blochwave(basis, scfres.ψ, δV, zero(Vec3{T}));
                    occupation_threshold=scfres.occupation_threshold, bandtolalg,
                    tol=tol / 10, maxiter=maxiter_sternheimer)
    end

    # 3. Tr[γ⁽¹⁾ H_b] = Σ_k w_k Σ_n [δocc_n ⟨ψ_n|H_b|ψ_n⟩ + 2 occ_n Re⟨δψ_n|H_b ψ_n⟩]
    #    Local part of H_b: ∫ (ρ̃ + δρ_γ) δV_loc, i.e. the local parts of B and of the trace
    #    together (δρ_γ = χ0 δV is the density of γ⁽¹⁾). Nonlocal part: atom-local projectors.
    @timing "nabec: local parts" begin
        δρ_γ = [compute_δρ(basis, scfres.ψ, responses[α].δψ, scfres.occupation,
                           responses[α].δoccupation;
                           occupation_threshold=scfres.occupation_threshold) for α = 1:3]
        inv_lattice = model.inv_lattice
        for ia in eachindex(atoms), β = 1:3
            δVβ = sum(inv_lattice[γ, β] .* bare.δVbare[γ, ia] for γ = 1:3)  # Cartesian β
            for α = 1:3
                Zel[ia][α, β] += sum((ρ̃[α] .+ δρ_γ[α])[:, :, :, 1] .* δVβ) * basis.dvol
            end
        end
    end
    @timing "nabec: trace with bare H" for (ik, kpt) in enumerate(basis.kpoints), ia in eachindex(atoms)
        ψk  = scfres.ψ[ik]
        occ = scfres.occupation[ik]
        Hτψ = apply_nonlocal_displacement_derivative_cart(basis, ik, atoms[ia], ψk)
        for α = 1:3, β = 1:3
            δocc, δψk = responses[α].δoccupation[ik], responses[α].δψ[ik]
            tr_γH = sum(δocc[n] * real(dot(ψk[:, n], Hτψ[β][:, n])) +
                        2 * occ[n] * real(dot(δψk[:, n], Hτψ[β][:, n]))
                        for n in axes(ψk, 2))
            Zel[ia][α, β] += basis.kweights[ik] * tr_γH
        end
    end
    Z_electronic = [mpi_sum(Z, basis.comm_kpts) for Z in Zel]
    Z_ionic = [charge_ionic(model.atoms[s]) * Matrix{T}(I, 3, 3) for s in atoms]
    (; Z=Z_ionic .+ Z_electronic, Z_electronic, Z_ionic, atoms)
end
