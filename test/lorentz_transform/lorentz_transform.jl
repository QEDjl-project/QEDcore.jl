using QEDcore
using Random

include("utils.jl")
include("../test_implementation/TestImplementation.jl")

const RNG = MersenneTwister(12345)
const ATOL = 1e-15

const test_mom = rand(RNG, SFourMomentum)
const test_psf = ParticleStateful(
    Incoming(), rand(RNG, TestImplementation.PARTICLE_SET), test_mom
)
const test_mass_square = test_mom * test_mom

const TESTMODEL = TestImplementation.TestModel()
const TESTPSDEF = TestImplementation.TestPhasespaceDef()

@testset "beta boost" begin
    @testset "defaults" begin
        xyz = rand(RNG, 3)
        xyz = @. (2 * xyz - 1) / sqrt(3)
        x, y, z = xyz
        boost_x_default = Boost(x)
        @test boost_x_default.param == BetaX(x)

        boost_vec_default = Boost(x, y, z)
        @test boost_vec_default.param == BetaVector(x, y, z)
    end

    @testset "$val_type" for val_type in (Float64, Float32)
        @testset "axis beta" begin
            @testset "$beta_param_type" for beta_param_type in (BetaX, BetaY, BetaZ)
                test_beta_val = 2 * rand(RNG, val_type) - 1
                test_beta = beta_param_type(test_beta_val)

                @testset "element type" begin
                    @test eltype(test_beta) == val_type
                end

                # test conversions
                for comp_val_type in (Float64, Float32)
                    comp_beta = beta_param_type(comp_val_type(test_beta_val))

                    @testset "convert element" begin
                        test_beta_after = beta_param_type{comp_val_type}(test_beta_val)
                        @test test_beta_after == comp_beta
                    end

                    @testset "convert type" begin
                        test_beta_after = convert(beta_param_type{comp_val_type}, test_beta)
                        @test test_beta_after == comp_beta
                    end
                end
            end
        end
    end

    @testset "$boost_param_type" for boost_param_type in (BetaVector, BetaX, BetaY, BetaZ)
        test_param = _rand(RNG, boost_param_type)
        boost = Boost(test_param)

        @testset "invariance" begin
            test_mom_prime = boost(test_mom)
            test_psf_prime = boost(test_psf)
            @test isapprox(test_mom_prime * test_mom_prime, test_mass_square)
            @test isapprox(
                momentum(test_psf_prime) * momentum(test_psf_prime), test_mass_square
            )
        end

        @testset "inversion" begin
            inv_boost_direct = Boost(-test_param)
            inv_boost = inv(boost)

            @test isapprox(inv_boost_direct(boost(test_mom)), test_mom)
            @test isapprox(inv_boost(boost(test_mom)), test_mom)
        end

        @testset "phase space point" begin
            test_param = _rand(RNG, boost_param_type)
            boost = Boost(test_param)
            @testset "($N_INCOMING,$N_OUTGOING)" for (N_INCOMING, N_OUTGOING) in
                                                     Iterators.product(
                (1, rand(RNG, 2:8)), (1, rand(RNG, 2:8))
            )
                INCOMING_PARTICLES = Tuple(
                    rand(RNG, TestImplementation.PARTICLE_SET, N_INCOMING)
                )
                OUTGOING_PARTICLES = Tuple(
                    rand(RNG, TestImplementation.PARTICLE_SET, N_OUTGOING)
                )

                TESTPROC = TestImplementation.TestProcess(
                    INCOMING_PARTICLES, OUTGOING_PARTICLES
                )
                IN_PS = TestImplementation._rand_momenta(RNG, N_INCOMING)
                OUT_PS = TestImplementation._rand_momenta(RNG, N_OUTGOING)
                PSP = PhaseSpacePoint(TESTPROC, TESTMODEL, TESTPSDEF, IN_PS, OUT_PS)

                PSP_prime = boost(PSP)
                @test isapprox(
                    [getMass2.(momenta(PSP, Incoming()))...],
                    [getMass2.(momenta(PSP_prime, Incoming()))...],
                )
                @test isapprox(
                    [getMass2.(momenta(PSP, Outgoing()))...],
                    [getMass2.(momenta(PSP_prime, Outgoing()))...],
                )
            end
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
