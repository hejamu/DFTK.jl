# Fast validation suite (low settings). Run on AMAM:  scripts/amam.sh -c 8 -m 16G -- test/runtests.jl
# The H2O FD comparison and the Al k-convergence scan are separate (test_water.jl,
# scripts/validate_al.jl): they take longer.
using Test

@testset "NABEC" begin
    include("test_velocity.jl")
    include("test_drude.jl")
    include("test_polar.jl")
    include("test_sternheimer.jl")
end
