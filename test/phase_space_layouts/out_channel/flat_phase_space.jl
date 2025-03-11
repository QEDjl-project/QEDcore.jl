using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestModel()
TESTINPSL = TestImplementation.TestInPhaseSpaceLayout()
N_INCOMING = 2
INCOMING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_INCOMING))
IN_MASSES = mass.(INCOMING_PARTICLES)
SUM_IN_MASSES = sum(IN_MASSES)

@testset "$N_OUTGOING" for N_OUTGOING in 2:10
    OUTGOING_PARTICLES = Tuple(rand(RNG, TestImplementation.PARTICLE_SET, N_OUTGOING))

    OUT_MASSES = mass.(OUTGOING_PARTICLES)
    SUM_OUT_MASSES = sum(OUT_MASSES)
    TESTSQRTS = (1 + rand(RNG)) * (SUM_OUT_MASSES + SUM_IN_MASSES)

    TESTPROC = TestImplementation.TestProcess(INCOMING_PARTICLES, OUTGOING_PARTICLES)

    TESTINMOMS = TestImplementation._generate_onshell_two_body_moms(
        RNG, IN_MASSES, TESTSQRTS
    )

    test_out_psl = FlatPhaseSpaceLayout(TESTINPSL)

    TESTOUTCOORDS = Tuple(rand(RNG, 4 * N_OUTGOING))

    @testset "properties" begin
        @test QEDbase.phase_space_dimension(TESTPROC, TESTMODEL, test_out_psl) ==
            4 * N_OUTGOING
        @test QEDbase.in_phase_space_layout(test_out_psl) == TESTINPSL
    end

    @testset "momentum generation" begin
        test_out_moms = QEDbase.build_momenta(
            TESTPROC, TESTMODEL, TESTINMOMS, test_out_psl, TESTOUTCOORDS
        )

        @test length(test_out_moms) == N_OUTGOING
        @test isapprox(getMass.(test_out_moms), [OUT_MASSES...], atol=ATOL, rtol=RTOL)
        @test isapprox(sum(test_out_moms), sum(TESTINMOMS), atol=ATOL, rtol=RTOL)
        @test test_out_moms isa Tuple
    end
end

@testset "error handling" begin
    valid_in_particles = (
        TestImplementation.TestParticleFermion(), TestImplementation.TestParticleFermion()
    )
    sum_in_masses = sum(mass.(valid_in_particles))
    valid_out_particles = (
        TestImplementation.TestParticleBoson(), TestImplementation.TestParticleBoson()
    )
    sum_out_masses = sum(mass.(valid_out_particles))

    valid_process = TestImplementation.TestProcess(valid_in_particles, valid_out_particles)
    test_out_psl = FlatPhaseSpaceLayout(TESTINPSL)

    valid_sqrt_s = (1 + rand(RNG)) * (+sum(mass.(valid_out_particles)))

    valid_in_moms = TestImplementation._generate_onshell_two_body_moms(
        RNG, mass.(valid_in_particles), valid_sqrt_s
    )

    valid_out_coords = Tuple(rand(RNG, 8))

    @testset "not enough particles" begin
        invalid_out_particles = (rand(RNG, TestImplementation.PARTICLE_SET),)
        invalid_process = TestImplementation.TestProcess(
            valid_in_particles, invalid_out_particles
        )
        out_coords = Tuple(rand(RNG, 4))
        @test_throws InvalidInputError QEDbase.build_momenta(
            invalid_process, TESTMODEL, valid_in_moms, test_out_psl, out_coords
        )
    end

    @testset "not enough energy" begin
        invalid_sqrt_s = (rand(RNG) * (sum_out_masses - sum_in_masses)) + sum_in_masses
        invalid_in_moms = TestImplementation._generate_onshell_two_body_moms(
            RNG, mass.(valid_in_particles), invalid_sqrt_s
        )

        @test_throws InvalidInputError QEDbase.build_momenta(
            valid_process, TESTMODEL, invalid_in_moms, test_out_psl, valid_out_coords
        )
    end
end
