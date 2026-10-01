using Pkg
Pkg.develop(path=expanduser("~/repos/DFTK.jl"))
Pkg.add(["ForwardDiff", "PseudoPotentialData", "LinearAlgebra", "Printf", "Test", "Unitful", "UnitfulAtomic"])
Pkg.precompile()
