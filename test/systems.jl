# Small test systems (LDA, GTH, no NLCC, no symmetries). Low cutoffs: fast, not converged.
using DFTK
using PseudoPotentialData
using LinearAlgebra

const GTH_LDA = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")

function silicon(; Ecut=10, kgrid=[2, 2, 2], a=10.26)
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, GTH_LDA)
    model = model_DFT(lattice, [Si, Si], [ones(3)/8, -ones(3)/8];
                      functionals=LDA(), symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""fcc Al, 1-atom primitive cell."""
function aluminium(; Ecut=10, kgrid=[4, 4, 4], a=7.60, temperature=0.01,
                   smearing=Smearing.Gaussian())
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Al = ElementPsp(:Al, GTH_LDA)
    model = model_DFT(lattice, [Al], [zeros(3)]; functionals=LDA(), symmetries=false,
                      temperature, smearing)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""fcc Al, 4-atom conventional cubic cell."""
function aluminium_conventional(; Ecut=10, kgrid=[3, 3, 3], a=7.60, temperature=0.01,
                                smearing=Smearing.Gaussian())
    Al = ElementPsp(:Al, GTH_LDA)
    positions = [[0, 0, 0.], [0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]]
    model = model_DFT(a * I(3), fill(Al, 4), positions; functionals=LDA(),
                      symmetries=false, temperature, smearing)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""
Isolated H2O (LDA geometry close to experiment: r_OH = 1.81 bohr, ∠HOH = 104.5°),
centered in a cubic box of side `L` bohr; slightly off-symmetric positions avoid
accidental degeneracies with the box.
"""
function water(; Ecut=20, L=12.0, kgrid=[1, 1, 1])
    r, θ = 1.81, deg2rad(104.5)
    c = [L/2, L/2, L/2]
    R_O  = c
    R_H1 = c + r * [ sin(θ/2), cos(θ/2), 0]
    R_H2 = c + r * [-sin(θ/2), cos(θ/2), 0]
    R_O  = R_O  - [0, 0.3, 0]   # move the center of charge closer to the box center
    R_H1 = R_H1 - [0, 0.3, 0]
    R_H2 = R_H2 - [0, 0.3, 0]
    lattice = L * I(3)
    O = ElementPsp(:O, GTH_LDA)
    H = ElementPsp(:H, GTH_LDA)
    positions = [lattice \ R for R in (R_O, R_H1, R_H2)]
    model = model_DFT(Matrix(lattice), [O, H, H], positions; functionals=LDA(),
                      symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""Rock-salt NaCl, 2-atom primitive cell."""
function nacl(; Ecut=15, kgrid=[3, 3, 3], a=10.40)
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Na = ElementPsp(:Na, GTH_LDA)
    Cl = ElementPsp(:Cl, GTH_LDA)
    model = model_DFT(lattice, [Na, Cl], [zeros(3), ones(3)/2]; functionals=LDA(),
                      symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

scf(basis; tol=1e-10, kwargs...) = self_consistent_field(basis; tol, kwargs...)
