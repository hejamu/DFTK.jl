# Dipole moment of an isolated system in a periodic box, for finite-difference APTs.

"""
    dipole_moment(basis, ρ; center)

``d = Σ_κ Z^ion_κ (R_κ - c) - ∫ (r - c) ρ(r) dr`` in e·bohr (electrons carry charge -1).
`center` (Cartesian) defaults to the box center. Only meaningful when the density is
well localized away from the cell boundary.
"""
function dipole_moment(basis::PlaneWaveBasis, ρ;
                       center=basis.model.lattice * Vec3(0.5, 0.5, 0.5))
    model = basis.model
    ρtot  = DFTK.total_density(ρ)
    rs    = r_vectors_cart(basis)
    d_el  = -sum(ρtot[i] .* (rs[i] - center) for i in eachindex(rs)) * basis.dvol
    d_ion = sum(charge_ionic(model.atoms[s]) * (model.lattice * model.positions[s] - center)
                for s in eachindex(model.positions))
    Vector(d_el + d_ion)
end
