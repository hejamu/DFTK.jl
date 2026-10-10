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

@testitem "NLCC: displaced core potential vs finite differences" #=
    =#    tags=[:dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    dojo = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    Na = ElementPsp(:Na, dojo); Cl = ElementPsp(:Cl, dojo)
    @test DFTK.has_core_density(Cl)   # Na has no model core in this family
    lattice = 10.40 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model(positions) = model_DFT(lattice, [Na, Cl], positions; functionals=LDA(),
                                 symmetries=false)
    positions = [zeros(3), [0.5, 0.48, 0.5]]
    basis = PlaneWaveBasis(model(positions); Ecut=15, kgrid=[1, 1, 1])
    ρ = guess_density(basis)
    ixc = findfirst(t -> t isa DFTK.TermXc, basis.terms)
    Vxc(b) = DFTK.xc_potential_real(b.terms[ixc], b, nothing, nothing; ρ).potential

    h = 1e-4
    @test isnothing(DFTK.xc_core_displacement_potential(basis.terms[ixc], basis, 1, 1; ρ))
    for (α, s) in ((1, 2), (3, 2))
        δV = DFTK.xc_core_displacement_potential(basis.terms[ixc], basis, α, s; ρ)
        Vh = map((h, -h)) do ε
            pos = deepcopy(positions)
            pos[s] = pos[s] + ε * Vector(I[1:3, α])
            Vxc(PlaneWaveBasis(model(pos); Ecut=15, kgrid=[1, 1, 1], basis.fft_size))
        end
        δV_fd = (Vh[1] - Vh[2]) / 2h
        @test norm(δV - δV_fd) < 1e-6 * norm(δV_fd)
    end
end

@testitem "NABEC with NLCC: translation identity of the screened potentials" #=
    =#    tags=[:dont_test_mpi] begin
    # Moving all atoms rigidly shifts the whole system, so per k-point
    #   Σ_s δV_s = −∂V_tot/∂x_α,   V_tot = V_loc + V_H + V_xc[ρ + ρ_core].
    # Without the displaced-core term this is violated at the 2e-2 level here; the remaining
    # error is the XC grid aliasing. (The acoustic sum rule Σ_s Z_s = 0 itself holds only for
    # converged k-meshes and is not a useful small test.)
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    dojo = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    Na = ElementPsp(:Na, dojo); Cl = ElementPsp(:Cl, dojo)
    lattice = 10.40 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model = model_DFT(lattice, [Na, Cl], [zeros(3), [0.5, 0.47, 0.52]]; functionals=LDA(),
                      symmetries=false)
    basis  = PlaneWaveBasis(model; Ecut=20, kgrid=[2, 2, 2])
    scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
    pot = DFTK.nabec_phonon_potentials(scfres; tol=1e-10)
    Vtot_fourier = fft(basis, DFTK.total_local_potential(scfres.ham)[:, :, :, 1])
    for α = 1:3
        mdV = real(irfft(basis, map((G, v) -> -2π * im * G[α] * v,
                                    G_vectors(basis), Vtot_fourier)))
        @test norm(sum(pot.δV[α, :]) - mdV) < 3e-3 * norm(mdV)
    end
end

@testitem "NABEC: field-response route agrees with phonon route" #=
    =#    tags=[:slow, :dont_test_mpi] setup=[NabecSystems] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData
    using .NabecSystems: silicon, aluminium

    dojo = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    Na = ElementPsp(:Na, dojo); Cl = ElementPsp(:Cl, dojo)
    lattice = 10.40 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    nacl = PlaneWaveBasis(model_DFT(lattice, [Na, Cl], [zeros(3), [0.5, 0.47, 0.52]];
                                    functionals=LDA(), symmetries=false);
                          Ecut=15, kgrid=[2, 2, 2])
    for (name, basis) in (("Si", silicon(; kgrid=[2, 2, 2])),
                          ("Al", aluminium(; kgrid=[4, 4, 4], temperature=0.05)),
                          ("NaCl (NLCC)", nacl))
        scfres = self_consistent_field(basis; tol=1e-11, callback=identity)
        Zph = compute_nabec(scfres; tol=1e-10).Z
        Zfd = compute_nabec_field(scfres; tol=1e-10).Z
        err = maximum(maximum(abs, a - b) for (a, b) in zip(Zph, Zfd))
        @info "field vs phonon route" name err
        @test err < 1e-5
    end
end

@testitem "NABEC: per-atom bare phonon derivatives vs automatic differentiation" #=
    =#    tags=[:dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    dojo = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    Na = ElementPsp(:Na, dojo); Cl = ElementPsp(:Cl, dojo)
    lattice = 10.40 / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    model = model_DFT(lattice, [Na, Cl, Cl], [zeros(3), [0.5, 0.47, 0.52], [0.21, 0.3, 0.77]];
                      functionals=LDA(), symmetries=false, temperature=0.01,
                      smearing=Smearing.Gaussian())     # odd electron count
    basis = PlaneWaveBasis(model; Ecut=12, kgrid=ExplicitKpoints([[0.1, 0.2, -0.15]]))
    scfres = self_consistent_field(basis; tol=1e-8, callback=identity)
    ψk = scfres.ψ[1]
    bare = DFTK.nabec_bare_potentials(scfres)
    xc = only(filter(t -> t isa DFTK.TermXc, basis.terms))
    for s in 1:3, α in 1:3
        # nonlocal: atom-local projectors vs AD through all projectors
        a = DFTK.apply_nonlocal_displacement_derivative(basis, 1, α, s, ψk)
        b = DFTK.apply_nonlocal_displacement_derivative_ad(basis, 1, α, s, ψk)
        @test norm(a - b) < 1e-10 * max(1, norm(b))
        # local: analytic structure-factor derivative vs AD of compute_local_potential
        Vad = DFTK.derivative_wrt_αs(model.positions, α, s) do pos
            DFTK.compute_local_potential(basis; positions=pos)
        end
        core = DFTK.xc_core_displacement_potential(xc, basis, α, s; ρ=scfres.ρ)
        isnothing(core) || (Vad = Vad .+ core[:, :, :, 1])
        @test norm(bare.δVbare[α, s] - Vad) < 1e-10 * max(1, norm(Vad))
    end
end
