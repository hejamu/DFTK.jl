# Phase 3, test 1: isolated H2O. The NABEC of an insulator is the adiabatic APT, so the
# velocity-gauge sum over states must agree with finite differences of the dipole moment
# from displaced-geometry SCFs. Also checks the acoustic sum rule Σ_κ Z_κ = 0.
using Test
using NABEC
using DFTK
using LinearAlgebra
using Printf

include("systems.jl")

function fd_apt(basis; h=1e-3, tol=1e-11)
    model = basis.model
    n_atoms = length(model.positions)
    Z = [zeros(3, 3) for _ = 1:n_atoms]
    for s = 1:n_atoms, β = 1:3
        d = map((+1, -1)) do sgn
            positions = copy(model.positions)
            shift = zeros(3); shift[β] = sgn * h
            positions[s] = positions[s] + model.inv_lattice * shift
            m = Model(model; positions)
            b = PlaneWaveBasis(m; basis.Ecut, kgrid=basis.kgrid, fft_size=basis.fft_size)
            res = self_consistent_field(b; tol, callback=identity)
            dipole_moment(b, res.ρ)
        end
        Z[s][:, β] = (d[1] - d[2]) / 2h
    end
    Z
end

function print_tensors(label, Zs)
    println(label)
    for (s, Z) in enumerate(Zs)
        println("  atom $s:")
        for α = 1:3
            @printf "    % .5f % .5f % .5f\n" Z[α, 1] Z[α, 2] Z[α, 3]
        end
    end
end

@testset "H2O: NABEC (SOS) = FD dipole APT" begin
    basis  = water(; Ecut=get(ENV, "NABEC_ECUT", "20") |> x -> parse(Float64, x),
                   L=get(ENV, "NABEC_L", "10") |> x -> parse(Float64, x),
                   kgrid=fill(parse(Int, get(ENV, "NABEC_NK", "2")), 3))
    scfres = self_consistent_field(basis; tol=1e-11)
    println("n_G = ", length(G_vectors(basis, basis.kpoints[1])))
    method = get(ENV, "NABEC_METHOD", "sos")
    t = @elapsed res = method == "sos" ? nabec_sos(scfres; tol_response=1e-10) :
                                         nabec_sternheimer(scfres; tol_response=1e-10)
    println("NABEC ($method) time: $t s")
    Zfd = fd_apt(basis)
    print_tensors("NABEC (velocity gauge, SOS):", res.Z)
    print_tensors("FD dipole APT:", Zfd)
    maxdiff = maximum(maximum(abs, res.Z[s] - Zfd[s]) for s in eachindex(Zfd))
    asr_sos = maximum(abs, sum(res.Z))
    asr_fd  = maximum(abs, sum(Zfd))
    @printf "L=%s nk=%s  max |Z_SOS - Z_FD| = %.2e   ASR residual: SOS %.2e, FD %.2e\n" get(ENV, "NABEC_L", "10") get(ENV, "NABEC_NK", "2") maxdiff asr_sos asr_fd
    @test maxdiff < 1e-3
    @test asr_sos < 2e-2
end
