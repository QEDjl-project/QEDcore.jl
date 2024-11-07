using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestModel()
TESTINPSL = TestImplementation.TrivialInPSL()
N_INCOMING = 2
INCOMING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_INCOMING))
IN_MASSES = mass.(INCOMING_PARTICLES)

#@testset "$N_OUTGOING" for N_OUTGOING in (2, rand(RNG,3:8))
@testset "$N_OUTGOING" for N_OUTGOING in 2:10
    OUTGOING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_OUTGOING))

    OUT_MASSES = mass.(OUTGOING_PARTICLES)
    SUM_OUT_MASSES = sum(OUT_MASSES)
    TESTSQRTS = (1 + rand(RNG)) * (SUM_OUT_MASSES + sum(IN_MASSES))

    TESTPROC = TestImplementation.TestProcess(INCOMING_PARTICLES, OUTGOING_PARTICLES)

    TESTINMOMS = TestImplementation._generate_onshell_two_body_moms(
        RNG, IN_MASSES, TESTSQRTS
    )

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

        @test length(test_out_moms) == N_OUTGOING
        @test isapprox(getMass.(test_out_moms), [OUT_MASSES...], atol=ATOL, rtol=RTOL)
        @test isapprox(sum(test_out_moms), sum(TESTINMOMS), atol=ATOL, rtol=RTOL)
    end
end
