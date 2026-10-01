# Au bulk and Au(111)/water slab (LDA, GTH semicore: Au with 11 e⁻).
using DFTK
using PseudoPotentialData
using LinearAlgebra

const GTH_LDA_SC = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
const DOJO_LDA   = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
const A_AU = 7.67   # bohr, ≈ LDA lattice constant of Au (4.06 Å)

# Pseudopotential family: :gth (cp2k GTH semicore, Au 11 e⁻) or :dojo (PseudoDojo NC, Au 19 e⁻).
# GTH projectors are too hard for converged velocity-type quantities (NOTES.md); PseudoDojo
# is used with NLCC switched off because DFTK's phonon δH does not include the core density.
family_of(f) = f == :dojo ? DOJO_LDA : GTH_LDA_SC
function lda_model(lattice, atoms, positions; family, temperature, smearing)
    if family == :dojo
        terms = [Kinetic(), AtomicLocal(), AtomicNonlocal(), Ewald(), PspCorrection(), Hartree(),
                 Xc([:lda_x, :lda_c_pw]; use_nlcc=false)]
        iszero(temperature) || push!(terms, Entropy())
        Model(lattice, atoms, positions; terms, symmetries=false, spin_polarization=:none,
              temperature, smearing)
    else
        model_DFT(lattice, atoms, positions; functionals=LDA(), symmetries=false,
                  temperature, smearing)
    end
end

"""fcc Au, 1-atom primitive cell."""
function gold(; Ecut=20, kgrid=[8, 8, 8], a=A_AU, temperature=0.03,
              smearing=Smearing.Gaussian(), family=:gth)
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Au = ElementPsp(:Au, family_of(family))
    model = lda_model(lattice, [Au], [zeros(3)]; family, temperature, smearing)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""
Cartesian atoms of an ABC Au(111) slab with `n_layers` (odd) layers, `n × n` lateral
supercell, centered at z = 0 on a middle-layer atom, and a flat H2O on a top site of the
upper surface plus its inversion image on the lower surface (so the slab is centrosymmetric,
no net dipole). Returns (lattice, symbols, cartesian positions).

- `h_O`: height of O above the top Au layer (bohr)
- `c`:   cell height (bohr)
- `with_gold=false` gives the same cell with the water molecules only (vacuum reference).
"""
function au111_water_geometry(; n=2, n_layers=3, a=A_AU, h_O=5.3, c=42.0, with_gold=true,
                              φ=deg2rad(30))
    @assert isodd(n_layers)
    d_nn  = a / √2
    d_z   = a / √3
    a1 = n * d_nn * [1.0, 0, 0]
    a2 = n * d_nn * [0.5, √3/2, 0]
    lattice = hcat(a1, a2, [0, 0, c])
    # ABC stacking offsets of the 1×1 cell (in units of the 1×1 surface vectors)
    offsets = ([0.0, 0.0], [1/3, 1/3], [2/3, 2/3])
    b1, b2 = d_nn * [1.0, 0, 0], d_nn * [0.5, √3/2, 0]
    mid = (n_layers + 1) ÷ 2
    symbols = Symbol[]
    R = Vector{Vector{Float64}}()
    top_site = zeros(3)
    for ℓ = 1:n_layers
        o = offsets[mod1(ℓ - mid + 2, 3)]   # middle layer gets offset B ([1/3,1/3])
        z = (ℓ - mid) * d_z
        for i = 0:n-1, j = 0:n-1
            r = (i + o[1]) * b1 + (j + o[2]) * b2 + [0, 0, z]
            with_gold && (push!(symbols, :Au); push!(R, r))
            (ℓ == n_layers && i == 0 && j == 0) && (top_site = r)
        end
    end
    # Shift so that the inversion center (a middle-layer atom, offset B) is at the origin
    center = (1/3) * b1 + (1/3) * b2
    R .= [r - center for r in R]
    top_site = top_site - center
    # Flat water, O on top of `top_site`, molecular plane parallel to the surface
    r_OH, θ = 1.81, deg2rad(104.5)
    O  = top_site + [0, 0, h_O]
    u1 = [cos(φ + θ/2), sin(φ + θ/2), 0]
    u2 = [cos(φ - θ/2), sin(φ - θ/2), 0]
    water = [(:O, O), (:H, O + r_OH * u1), (:H, O + r_OH * u2)]
    for (s, r) in water          # upper molecule
        push!(symbols, s); push!(R, r)
    end
    for (s, r) in water          # inversion image
        push!(symbols, s); push!(R, -r)
    end
    (; lattice, symbols, R)
end

function au111_water(; Ecut=20, kgrid=[4, 4, 1], temperature=0.03,
                     smearing=Smearing.Gaussian(), with_gold=true, family=:gth, kwargs...)
    (; lattice, symbols, R) = au111_water_geometry(; with_gold, kwargs...)
    if !with_gold    # water-only reference: an insulator
        temperature, smearing = 0.0, Smearing.None()
    end
    atoms = [ElementPsp(s, family_of(family)) for s in symbols]
    # Slab centered at z = c/2, so the vacuum straddles the cell boundary (dipoles along z
    # are then well defined with the default box-center origin of `dipole_moment`).
    c = lattice[3, 3]
    positions = [mod.(lattice \ (r + [0, 0, c/2]), 1.0) for r in R]
    model = lda_model(lattice, atoms, positions; family, temperature, smearing)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""Indices of the atoms of the upper water molecule (O, H, H) in `au111_water`."""
upper_water_indices(basis) = begin
    n_Au = count(a -> element_symbol(a) == :Au, basis.model.atoms)
    n_Au .+ (1:3)
end
