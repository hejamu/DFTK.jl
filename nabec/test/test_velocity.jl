# Phase 1: the velocity operator ∂H_k/∂k_α.
using Test
using NABEC
using DFTK
using LinearAlgebra

include("systems.jl")

# (a) Full matrix vs central finite difference of H_k(k±h) on the fixed G set.
function check_matrix_fd(scfres; ik=2, h=1e-4)
    basis = scfres.basis
    v = VelocityOperator(basis, ik)
    kred = basis.kpoints[ik].coordinate
    map(1:3) do α
        dk = basis.model.recip_lattice \ Vec3(ntuple(i -> Float64(i == α), 3))
        Hp = NABEC.hamiltonian_at_shifted_k(scfres.ham, ik, kred + h * dk)
        Hm = NABEC.hamiltonian_at_shifted_k(scfres.ham, ik, kred - h * dk)
        Vfd = (Hp - Hm) / 2h
        Vα  = NABEC.velocity_matrix(v, α)
        norm(Vα - Vfd) / norm(Vfd)
    end
end

# (b) Band velocities ⟨n|v|n⟩ vs finite differences of ε_nk from independent DFTK bases at
#     k ± h (fixed SCF density, non-degenerate bands only).
function check_band_velocity(scfres; ik=2, h=1e-4, n_bands=8)
    basis = scfres.basis
    model = basis.model
    kred  = basis.kpoints[ik].coordinate
    ham0  = scfres.ham
    ε0    = eigen(Hermitian(Matrix(Array(ham0.blocks[ik]))))
    v     = VelocityOperator(basis, ik)
    errors = Float64[]
    for α = 1:3
        dk = model.recip_lattice \ Vec3(ntuple(i -> Float64(i == α), 3))
        εpm = map((+1, -1)) do sgn
            b = PlaneWaveBasis(model; Ecut=basis.Ecut,
                               kgrid=ExplicitKpoints([kred + sgn * h * dk]),
                               fft_size=basis.fft_size)
            ham = Hamiltonian(b; ρ=scfres.ρ)
            eigvals(Hermitian(Matrix(Array(ham.blocks[1]))))[1:n_bands]
        end
        vfd = (εpm[1] - εpm[2]) / 2h
        for n = 1:n_bands
            ε = ε0.values
            gap = min(n > 1 ? ε[n] - ε[n-1] : Inf, ε[n+1] - ε[n])
            gap < 1e-3 && continue
            ψn = ε0.vectors[:, n]
            vn = real(dot(ψn, apply_velocity(v, α, ψn)))
            push!(errors, abs(vn - vfd[n]))
        end
    end
    maximum(errors)
end

@testset "velocity operator" begin
    for (name, basis) in [("Si", silicon()), ("Al", aluminium(kgrid=[2, 2, 2]))]
        scfres = scf(basis; tol=1e-8)
        @testset "$name: matrix vs FD" begin
            errs = check_matrix_fd(scfres)
            println("$name  ‖v - v_FD‖/‖v_FD‖ = ", errs)
            @test all(errs .< 1e-7)
        end
        @testset "$name: band velocities vs FD" begin
            err = check_band_velocity(scfres)
            println("$name  max |⟨n|v|n⟩ - dε/dk| = ", err)
            @test err < 1e-6
        end
    end
    @testset "nonlocal part vanishes for local-only potentials" begin
        Si = ElementCohenBergstresser(:Si)
        lattice = Si.lattice_constant / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
        model = Model(lattice, [Si, Si], [ones(3)/8, -ones(3)/8];
                      terms=[Kinetic(), AtomicLocal()], symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=8, kgrid=[2, 2, 2])
        v = VelocityOperator(basis, 2)
        @test isnothing(v.P)
        kpG = Gplusk_vectors_cart(basis, basis.kpoints[2])
        @test NABEC.velocity_matrix(v, 1) ≈ Diagonal([p[1] for p in kpG])
    end
end
