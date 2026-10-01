# # Nonadiabatic Born effective charges
#
# The Born effective charge ``Z^*_{κ,αβ}`` of an insulator is the change of polarization
# (or of the force on atom ``κ``) under a displacement (or a field). In metals the static
# definition breaks down because free carriers screen any static field. Dreyer, Coh and
# Stengel ([PRL **128**, 095901 (2022)](https://doi.org/10.1103/PhysRevLett.128.095901))
# define instead *nonadiabatic* Born effective charges (NABECs) as the current response to
# an atomic velocity, taking the long-wavelength limit before the static limit. The
# intraband (Drude) term is then excluded, and the charges satisfy the sum rule
# ```math
# \sum_κ Z^*_{κ,αβ} = \frac{Ω}{π} \tilde D_{αβ},
# ```
# where ``\tilde D`` is the Drude weight evaluated with the canonical momentum (with
# nonlocal pseudopotentials it differs slightly from the usual Drude weight ``D``).
# For insulators NABECs equal the ordinary Born effective charges.

# !!! warning "Preliminary implementation"
#     The same limitations as for [Phonon computations](@ref) apply: symmetries must be
#     disabled, only LDA has been tested and non-linear core corrections are not supported.
#     Velocity-type quantities need pseudopotentials that converge well in plane waves;
#     very hard projectors (e.g. GTH for oxygen) converge extremely slowly with `Ecut`.
#     Metals need ``k``-meshes that resolve the smearing.
#
# We check the sum rule for fcc aluminium with a large smearing, so that a modest
# ``k``-mesh suffices.

using DFTK
using LinearAlgebra
using PseudoPotentialData

pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
a = 7.60  # bohr
lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Al = ElementPsp(:Al, pseudopotentials)
model = model_DFT(lattice, [Al], [zeros(3)]; functionals=LDA(), symmetries=false,
                  temperature=0.1, smearing=Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut=8, kgrid=[8, 8, 8])
scfres = self_consistent_field(basis; tol=1e-10);

# The NABEC tensors (one ``3×3`` matrix per atom; first index: current direction, second:
# displacement direction):

nabec = compute_nabec(scfres; tol=1e-10)
nabec.Z[1]

# and the Drude weights ``(Ω/π)D`` and ``(Ω/π)\tilde D`` in electrons per cell:

drude = compute_drude_weight(scfres)
(; Z=nabec.Z[1][1, 1], D=drude.D[1, 1], Dtilde=drude.Dtilde[1, 1])

# The NABEC of the single atom matches ``(Ω/π)\tilde D`` (here to a few ``10^{-3}``),
# while ``(Ω/π)D`` differs by the nonlocal pseudopotential correction.
