using QEDcore
using Random

include("utils.jl")

const RNG = MersenneTwister(12345)
const ATOL = 1e-15

const test_mom = rand(RNG, SFourMomentum)
const test_mass_square = test_mom * test_mom

@testset "beta boost" begin
    @testset "defaults" begin
        x, y, z = rand(RNG, 3)
        boost_x_default = Boost(x)
        @test boost_x_default.param == BetaX(x)

        boost_vec_default = Boost(x, y, z)
        @test boost_vec_default.param == BetaVector(x, y, z)
    end

    @testset "$boost_type" for boost_type in (BetaVector, BetaX, BetaY, BetaZ)
        test_param = _rand(RNG, boost_type)
        boost = Boost(test_param)

        @testset "invariance" begin
            test_mom_prime = boost(test_mom)
            @test isapprox(test_mom_prime * test_mom_prime, test_mass_square)
        end

        @testset "inversion" begin
            inv_boost_direct = Boost(-test_param)
            inv_boost = inv(boost)

            @test isapprox(inv_boost_direct(boost(test_mom)), test_mom)
            @test isapprox(inv_boost(boost(test_mom)), test_mom)
        end
    end

    @testset "recover axis boost" begin
        test_beta_vec = _rand_beta(RNG, Float64)
        boost = Boost(test_beta_vec)

        @testset "x boost" begin
            rnd_beta = rand(RNG)
            beta_x = BetaX(rnd_beta)
            beta_vec_x = BetaVector(rnd_beta, 0.0, 0.0)

            boost_axis = Boost(beta_x)
            boost_vec = Boost(beta_vec_x)

            @test isapprox(boost_axis(test_mom), boost_vec(test_mom))
        end
        @testset "y boost" begin
            rnd_beta = rand(RNG)
            beta_y = BetaY(rnd_beta)
            beta_vec_y = BetaVector(0.0, rnd_beta, 0.0)

            boost_axis = Boost(beta_y)
            boost_vec = Boost(beta_vec_y)

            @test isapprox(boost_axis(test_mom), boost_vec(test_mom))
        end
        @testset "z boost" begin
            rnd_beta = rand(RNG)
            beta_z = BetaZ(rnd_beta)
            beta_vec_z = BetaVector(0.0, 0.0, rnd_beta)

            boost_axis = Boost(beta_z)
            boost_vec = Boost(beta_vec_z)

            @test isapprox(boost_axis(test_mom), boost_vec(test_mom))
        end
    end
end
