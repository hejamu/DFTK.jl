using NABEC, DFTK, LinearAlgebra, Printf
lattice = 7.60 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
for (T, nk, full) in [(0.02, 12, false), (0.02, 12, true), (0.02, 20, false), (0.04, 12, false), (0.04, 20, false), (0.01, 24, false)]
    model = Model(lattice; n_electrons=3, terms=[Kinetic()], symmetries=false,
                  temperature=T, smearing=Smearing.Gaussian(), spin_polarization=:none)
    basis = PlaneWaveBasis(model; Ecut=4, kgrid=[nk, nk, nk])
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
    bands = full ? full_bands(scfres) : scf_bands(scfres)
    D = drude_weight(scfres, bands).D
    @printf "T=%.2f nk=%2d full=%d  ΩD/π=%.5f\n" T nk full D[1,1]
end
