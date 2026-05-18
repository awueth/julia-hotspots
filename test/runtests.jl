using Test

module LSERegressionTests
    include("test_lse_regression.jl")
end

module PotentialGeneratorTests
    include("test_potential_generator.jl")
end

module PotentialInterfaceTests
    include("test_potential_interface.jl")
end

module SpectralGalerkinTests
    include("test_spectral_galerkin.jl")
end
