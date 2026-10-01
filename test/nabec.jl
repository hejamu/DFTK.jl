@testmodule NabecSystems begin
using DFTK
using LinearAlgebra
using PseudoPotentialData

const gth = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
fcc(a) = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]

function silicon(; Ecut=10, kgrid=[2, 2, 2], kinetic_blowup=BlowupIdentity())
    Si = ElementPsp(:Si, gth)
    model = model_DFT(fcc(10.26), [Si, Si], [ones(3)/8, -ones(3)/8]; functionals=LDA(),
                      symmetries=false, kinetic_blowup)
    PlaneWaveBasis(model; Ecut, kgrid)
end
function aluminium(; Ecut=8, kgrid=[3, 3, 3], temperature=0.03)
    Al = ElementPsp(:Al, gth)
    model = model_DFT(fcc(7.60), [Al], [zeros(3)]; functionals=LDA(), symmetries=false,
                      temperature, smearing=Smearing.Gaussian())
    PlaneWaveBasis(model; Ecut, kgrid)
end
function nacl(; Ecut=10, kgrid=[2, 2, 2])
    Na = ElementPsp(:Na, gth); Cl = ElementPsp(:Cl, gth)
    model = model_DFT(fcc(10.40), [Na, Cl], [zeros(3), ones(3)/2]; functionals=LDA(),
                      symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

# Dense block of H_k for the model at reduced k-point `kcoord`, built with DFTK's own
# k-point machinery on the density ρ.
function hamiltonian_matrix(model, Ecut, fft_size, kcoord, ρ)
    b = PlaneWaveBasis(model; Ecut, kgrid=ExplicitKpoints([kcoord]), fft_size)
    (; G=b.kpoints[1].G_vectors, H=Matrix(Array(Hamiltonian(b; ρ).blocks[1])))
end
end


@testitem "Velocity operator: comparison to finite differences" #=
    =#    tags=[:dont_test_mpi] setup=[NabecSystems] begin
    using DFTK
    using LinearAlgebra
    using .NabecSystems: silicon, aluminium, hamiltonian_matrix

    k0 = [0.13, -0.21, 0.27]
    # The blown-up dispersion is steep near the cutoff: its central difference has an
    # O(h²) error of ≈ 5e-5 at h = 1e-4, so use a smaller step there.
    for (basis0, h, rtol) in ((silicon(), 1e-4, 1e-7), (aluminium(), 1e-4, 1e-7),
                              (silicon(; kinetic_blowup=BlowupCHV()), 1e-5, 1e-5))
        model = basis0.model
        scfres = self_consistent_field(basis0; tol=1e-8, callback=identity)
        # Basis at the general k-point k0, same density
        basis = PlaneWaveBasis(model; basis0.Ecut, kgrid=ExplicitKpoints([k0]),
                               fft_size=basis0.fft_size)
        n_G = length(G_vectors(basis, basis.kpoints[1]))
        v = DFTK.velocity_operators(basis, 1)
        for α = 1:3
            δk = model.recip_lattice \ [i == α ? 1.0 : 0.0 for i = 1:3]
            Hp = hamiltonian_matrix(model, basis0.Ecut, basis0.fft_size, k0 + h * δk, scfres.ρ)
            Hm = hamiltonian_matrix(model, basis0.Ecut, basis0.fft_size, k0 - h * δk, scfres.ρ)
            @assert Hp.G == Hm.G == basis.kpoints[1].G_vectors  # same G set: comparable
            Vfd = (Hp.H - Hm.H) / 2h
            Vα  = v[α](Matrix{ComplexF64}(I, n_G, n_G))
            @test norm(Vα - Vfd) / norm(Vfd) < rtol
        end
    end
end

@testitem "NABEC: Sternheimer agrees with sum over states" #=
    =#    tags=[:dont_test_mpi] setup=[NabecSystems] begin
    using DFTK
    using .NabecSystems: aluminium, nacl

    for basis in (aluminium(), nacl())
        scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
        potentials = DFTK.nabec_phonon_potentials(scfres; tol=1e-10)
        stn = compute_nabec(scfres; potentials, tol=1e-10)
        sos = DFTK.compute_nabec_sos(scfres; potentials)
        @test maximum(maximum(abs, a - b) for (a, b) in zip(stn.Z, sos.Z)) < 1e-6
    end
end

@testitem "Drude weight: free electrons" #=
    =#    tags=[:slow, :dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    lattice = 7.60 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model = Model(lattice; n_electrons=3, terms=[Kinetic()], symmetries=false,
                  temperature=0.04, smearing=Smearing.Gaussian(), spin_polarization=:none)
    basis = PlaneWaveBasis(model; Ecut=4, kgrid=[20, 20, 20])  # LOBPCG needs n_G ≫ n_bands
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
    (; D, Dtilde) = compute_drude_weight(scfres)
    @test D ≈ Dtilde                       # no nonlocal term
    @test diag(D) ≈ fill(3.0, 3) rtol=1e-2  # (Ω/π) D = N_el
end

@testitem "NABEC: metal sum rule Σ_κ Z_κ = (Ω/π) D̃" #=
    =#    tags=[:slow, :dont_test_mpi] setup=[NabecSystems] begin
    using DFTK
    using LinearAlgebra
    using .NabecSystems: aluminium

    scfres = self_consistent_field(aluminium(; kgrid=[8, 8, 8], temperature=0.1);
                                   tol=1e-10, callback=identity)
    (; Z) = compute_nabec(scfres; tol=1e-10)
    (; D, Dtilde) = compute_drude_weight(scfres)
    @test abs(Z[1][1, 1] - Dtilde[1, 1]) < 5e-3
    @test abs(Z[1][1, 1] - D[1, 1]) > 5e-2     # the nonlocal correction is visible for Al
    @test maximum(abs, Z[1] - Z[1][1, 1] * I) < 1e-6
end
