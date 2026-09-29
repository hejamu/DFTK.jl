# Velocity operator v_α = ∂H_k/∂k_α (Cartesian α, Hartree atomic units).
#
# With the G-vector set of the k-point held fixed:
#   kinetic:  ∂_kα ½|k+G|² = (k+G)_α                         (diagonal in G)
#   nonlocal: ∂_kα (P D P†) = (∂P) D P† + P D (∂P)†
# Local, Hartree and xc terms carry no explicit k dependence.
#
# P[G,i] = e^{-i(k+G)·R_i} f_i(k+G) / √Ω. The k dependence of the structure factor
# cancels between bra and ket, so differentiating the full P (as done here via
# ForwardDiff on the k-point coordinate) or only the form factor f_i gives the same
# operator. See NOTES.md, "Velocity operator".

struct VelocityOperator{T, TP}
    kpG_cart::Vector{Vec3{T}}   # (k+G) in Cartesian coordinates
    scaling::T                  # kinetic scaling factor
    P::TP                       # nonlocal projectors at k (nothing if no nonlocal term)
    dP::Vector{TP}              # ∂P/∂k_α for α = 1:3
    D::Matrix{T}                # nonlocal coupling coefficients
end

function kinetic_term(basis)
    terms = [t for t in basis.terms if t isa DFTK.TermKinetic]
    @assert length(terms) == 1
    only(terms)
end
function nonlocal_term(basis)
    terms = [t for t in basis.terms if t isa DFTK.TermAtomicNonlocal]
    isempty(terms) ? nothing : only(terms)
end

function check_kinetic_supported(model)
    kin = only(t for t in model.term_types if t isa Kinetic)
    kin.blowup isa BlowupIdentity || error(
        "Velocity operator implemented only for the standard kinetic energy (no blow-up)")
    kin
end

"""
Nonlocal projectors ``P[G,i] = e^{-i(k+G)·R_i} f_i(k+G)/√Ω`` for an arbitrary (possibly
ForwardDiff dual) reduced k-point coordinate `kcoord`, on the fixed G-vector set `Gvecs`.
Mirrors `DFTK.build_projection_vectors`, but lets the element type follow `kcoord`.
"""
function nonlocal_projectors(basis::PlaneWaveBasis, kcoord::Vec3{TK}, Gvecs) where {TK}
    model = basis.model
    psp_groups = [group for group in model.atom_groups
                  if model.atoms[first(group)] isa ElementPsp]
    psps = [model.atoms[first(group)].psp for group in psp_groups]
    psp_positions = [model.positions[group] for group in psp_groups]

    Gpk      = [G + kcoord for G in Gvecs]
    Gpk_cart = [model.recip_lattice * p for p in Gpk]
    n_proj   = DFTK.count_n_proj(psps, psp_positions)
    P = zeros(Complex{TK}, length(Gvecs), n_proj)
    offset = 0
    for (psp, positions) in zip(psps, psp_positions)
        form_factors = DFTK.build_projector_form_factors(psp, Gpk_cart)
        for r in positions
            structure_factors = [cis(-2π * dot(p, r)) for p in Gpk]
            for iproj = 1:DFTK.count_n_proj(psp)
                P[:, offset+iproj] .= structure_factors .* form_factors[:, iproj] ./
                                      sqrt(model.unit_cell_volume)
            end
            offset += DFTK.count_n_proj(psp)
        end
    end
    @assert offset == n_proj
    P
end

"""
∂P/∂k_α (Cartesian α) at the k-point `kpt`, with the G-vector set fixed.

ForwardDiff through `norm(k+G)` yields NaN at k+G = 0 (a 0·∞ in the chain rule of the
radial part). Those rows are replaced by a central finite difference, which is accurate
to O(h²) because the projectors are smooth in k+G.
"""
function nonlocal_projector_derivative(basis::PlaneWaveBasis{T}, kpt, α; h=T(1e-5)) where {T}
    Gvecs  = DFTK.to_cpu(kpt.G_vectors)
    # Reduced displacement corresponding to a unit Cartesian displacement along α
    dk_red = basis.model.recip_lattice \ Vec3{T}(ntuple(i -> i == α, 3))
    dP = ForwardDiff.derivative(zero(T)) do ε
        nonlocal_projectors(basis, kpt.coordinate + ε * dk_red, Gvecs)
    end
    bad_rows = findall(i -> any(isnan, @view dP[i, :]), 1:size(dP, 1))
    if !isempty(bad_rows)
        Pp = nonlocal_projectors(basis, kpt.coordinate + h * dk_red, Gvecs)
        Pm = nonlocal_projectors(basis, kpt.coordinate - h * dk_red, Gvecs)
        dP[bad_rows, :] .= (Pp[bad_rows, :] .- Pm[bad_rows, :]) ./ 2h
    end
    dP
end

function VelocityOperator(basis::PlaneWaveBasis{T}, ik::Integer) where {T}
    kin = check_kinetic_supported(basis.model)
    kpt = basis.kpoints[ik]
    kpG_cart = Vector(Gplusk_vectors_cart(basis, kpt))

    nl = nonlocal_term(basis)
    if isnothing(nl) || size(nl.ops[ik].P, 2) == 0
        return VelocityOperator(kpG_cart, T(kin.scaling_factor), nothing,
                                Nothing[nothing, nothing, nothing], zeros(T, 0, 0))
    end
    P  = Matrix(nl.ops[ik].P)
    D  = Matrix(nl.ops[ik].D)
    dP = [nonlocal_projector_derivative(basis, kpt, α) for α = 1:3]
    VelocityOperator(kpG_cart, T(kin.scaling_factor), P, dP, D)
end

"""
Apply the canonical momentum ``p_α + k_α`` (the kinetic part of the velocity).
"""
function apply_canonical(v::VelocityOperator, α, ψ)
    v.scaling .* [p[α] for p in v.kpG_cart] .* ψ
end

"""
Apply the full velocity operator ``∂H/∂k_α`` to the columns of ψ.
"""
function apply_velocity(v::VelocityOperator, α, ψ)
    vψ = apply_canonical(v, α, ψ)
    if !isnothing(v.P)
        vψ .+= v.dP[α] * (v.D * (v.P' * ψ)) .+ v.P * (v.D * (v.dP[α]' * ψ))
    end
    vψ
end

"""
Dense matrix of ``∂H/∂k_α`` in the plane-wave basis of k-point `ik` (testing only).
"""
function velocity_matrix(v::VelocityOperator, α)
    apply_velocity(v, α, Matrix{complex(eltype(v.D))}(I, length(v.kpG_cart), length(v.kpG_cart)))
end

"""
Dense Hamiltonian H_k(k') built on the fixed G set of k-point `ik`, with the local
potential of `ham` but the kinetic and nonlocal parts evaluated at the reduced coordinate
`kcoord`. Used to test the velocity operator by finite differences.
"""
function hamiltonian_at_shifted_k(ham, ik, kcoord)
    basis = ham.basis
    kpt   = basis.kpoints[ik]
    Hk    = Matrix(Array(ham.blocks[ik]))
    kin   = kinetic_term(basis)
    nl    = nonlocal_term(basis)
    # Remove the kinetic and nonlocal parts at k ...
    Hk .-= Diagonal(kin.kinetic_energies[ik])
    if !isnothing(nl)
        Hk .-= Matrix(nl.ops[ik].P * (nl.ops[ik].D * nl.ops[ik].P'))
    end
    # ... and add them back at k'
    kcoord = Vec3(kcoord)
    scaling = kinetic_term(basis).scaling_factor
    kpG = [basis.model.recip_lattice * (G + kcoord) for G in DFTK.to_cpu(kpt.G_vectors)]
    Hk .+= Diagonal(scaling .* norm.(kpG) .^ 2 ./ 2)
    if !isnothing(nl)
        P = nonlocal_projectors(basis, kcoord, DFTK.to_cpu(kpt.G_vectors))
        Hk .+= P * (Matrix(nl.ops[ik].D) * P')
    end
    Hk
end
