using Random
using QEDbase
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("test_implementation/TestImplementation.jl")

@testset "($N_INCOMING,$N_OUTGOING)" for (N_INCOMING, N_OUTGOING) in Iterators.product(
    (1, rand(RNG, 2:8)), (1, rand(RNG, 2:8))
)
    INCOMING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_INCOMING))
    OUTGOING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_OUTGOING))

    TESTPROC = TestImplementation.TestProcess(INCOMING_PARTICLES, OUTGOING_PARTICLES)
    TESTMODEL = TestImplementation.TestModel()

    TESTINPSL = TestImplementation.TestInPhaseSpaceLayout()
    TESTINCOORDS = Tuple(rand(RNG, 4 * N_INCOMING))
    groundtruth_in_moms = TestImplementation._groundtruth_in_moms(TESTINCOORDS)

    TESTOUTPSL = TestImplementation.TestOutPhaseSpaceLayout(TESTINPSL)
    TESTOUTCOORDS = Tuple(rand(RNG, 4 * N_OUTGOING - 4))
    groundtruth_out_moms = TestImplementation._groundtruth_out_moms(
        groundtruth_in_moms, TESTOUTCOORDS
    )

    @testset "building directly" begin
        in_cmap = @inferred CoordinateMap(TESTPROC, TESTMODEL, TESTINPSL)
        out_cmap = @inferred CoordinateMap(TESTPROC, TESTMODEL, TESTOUTPSL)
        @testset "in channel" begin
            test_in_moms = @inferred in_cmap(TESTINCOORDS)
            @test all(isapprox.(test_in_moms, groundtruth_in_moms, atol=ATOL, rtol=RTOL))
        end

        @testset "out channel" begin
            test_in_moms, test_out_moms = @inferred out_cmap(TESTINCOORDS, TESTOUTCOORDS)
            @test all(isapprox.(test_in_moms, groundtruth_in_moms, atol=ATOL, rtol=RTOL))
            @test all(isapprox.(test_out_moms, groundtruth_out_moms, atol=ATOL, rtol=RTOL))
        end

        @testset "Error handling" begin
            @test_throws InvalidInputError in_cmap(TESTINCOORDS[2:end])
            @test_throws InvalidInputError in_cmap((TESTINCOORDS..., rand(RNG)))

            if N_OUTGOING != 1
                @test_throws InvalidInputError out_cmap(TESTINCOORDS, TESTOUTCOORDS[2:end])
                @test_throws InvalidInputError out_cmap(
                    TESTINCOORDS[2:end], TESTOUTCOORDS[2:end]
                )
                @test_throws InvalidInputError out_cmap(
                    (TESTINCOORDS..., rand(RNG)), TESTOUTCOORDS[2:end]
                )
            end
            @test_throws InvalidInputError out_cmap(
                TESTINCOORDS, (TESTOUTCOORDS..., rand(RNG))
            )
            @test_throws InvalidInputError out_cmap(
                TESTINCOORDS[2:end], (TESTOUTCOORDS..., rand(RNG))
            )
            @test_throws InvalidInputError out_cmap(
                (TESTINCOORDS..., rand(RNG)), (TESTOUTCOORDS..., rand(RNG))
            )
        end
    end

    @testset "building with cache" begin
        in_ccmap = @inferred CoordinateMapCached(
            TESTPROC, TESTMODEL, TESTINPSL, TESTINCOORDS
        )
        out_ccmap = @inferred CoordinateMapCached(
            TESTPROC, TESTMODEL, TESTOUTPSL, TESTINCOORDS
        )

        @testset "in channel" begin
            test_in_moms = @inferred in_ccmap()
            @test all(isapprox.(test_in_moms, groundtruth_in_moms, atol=ATOL, rtol=RTOL))
        end

        @testset "out channel" begin
            test_out_moms = @inferred out_ccmap(TESTOUTCOORDS)
            @test all(isapprox.(test_out_moms, groundtruth_out_moms, atol=ATOL, rtol=RTOL))
        end

        @testset "Error handling" begin
            @test_throws InvalidInputError CoordinateMapCached(
                TESTPROC, TESTMODEL, TESTINPSL, TESTINCOORDS[2:end]
            )
            @test_throws InvalidInputError CoordinateMapCached(
                TESTPROC, TESTMODEL, TESTINPSL, (TESTINCOORDS..., rand(RNG))
            )

            if N_OUTGOING != 1
                @test_throws InvalidInputError out_ccmap(TESTOUTCOORDS[2:end])
            end
            @test_throws InvalidInputError out_ccmap((TESTOUTCOORDS..., rand(RNG)))
        end
    end
end
