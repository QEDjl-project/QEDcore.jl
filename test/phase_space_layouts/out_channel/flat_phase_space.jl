using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestModel()
TESTINPSL = TestImplementation.TrivialInPSL()

@testset "($N_INCOMING,$N_OUTGOING)" for (N_INCOMING, N_OUTGOING) in Iterators.product(
    (1, rand(RNG, 2:8)), (1, rand(RNG, 2:8))
)
    INCOMING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_INCOMING))
    OUTGOING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_OUTGOING))

    TESTPROC = TestImplementation.TestProcess(INCOMING_PARTICLES, OUTGOING_PARTICLES)

    # TODO: generate onshell momenta
    TESTINCOORDS = Tuple(rand(RNG, 4 * N_INCOMING))
    TESTINMOMS = build_momenta(TESTPROC, TESTMODEL, TESTINPSL, TESTINCOORDS)

    test_out_psl = FlatPhaseSpaceLayout(TESTINPSL)

    TESTOUTCOORDS = Tuple(rand(RNG, 4 * N_OUTGOING))

    @testset "properties" begin
        @test phase_space_dimension(TESTPROC, TESTMODEL, test_out_psl) == 4 * N_OUTGOING
        @test in_phase_space_layout(test_out_psl) == TESTINPSL
    end

    @testset "momentum generation" begin
        test_out_moms = build_momenta(
            TESTPROC, TESTMODEL, TESTINMOMS, test_out_psl, TESTOUTCOORDS
        )

        @test length(test_out_psl) == N_OUTGOING
    end
end
