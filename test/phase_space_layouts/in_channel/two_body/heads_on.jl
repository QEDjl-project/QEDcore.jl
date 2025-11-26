using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestPerturbativeModel()

N_OUTGOING = 2
OUTPARTICLE = Tuple(rand(TestImplementation.PARTICLE_SET, N_OUTGOING))

TEST_AXES = (XAxis(), YAxis(), ZAxis())

_rho_from_E(E, m) = sqrt((E - m) * (E + m))

@testset "General" begin

    PARTICLE1, PARTICLE2 = Tuple(rand(TestImplementation.PARTICLE_SET, 2))

    TESTPROC = TestImplementation.TestProcess((PARTICLE1, PARTICLE2), OUTPARTICLE)

    ENERGIES1 = mass(PARTICLE1) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))
    ENERGIES2 = mass(PARTICLE2) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))

    @testset "$AXIS" for AXIS in TEST_AXES
        PSL = HeadsOnSystem(AXIS)
        coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)
        @testset "E1 = $E1, E2 = $E2" for (E1, E2) in Iterators.product(ENERGIES1, ENERGIES2)
            MOM1, MOM2 = @inferred coord_map((E1, E2))

            rho1 = _rho_from_E(E1, mass(PARTICLE1))
            rho2 = _rho_from_E(E2, mass(PARTICLE2))

            @test isapprox(getMass(MOM1), mass(PARTICLE1))
            @test isapprox(getMass(MOM2), mass(PARTICLE2))
            @test isapprox(getE(MOM1), E1)
            @test isapprox(getE(MOM2), E2)
            @test isapprox(-MOM1[2:4] ./ rho1, MOM2[2:4] ./ rho2)

        end
    end
end

@testset "Center-of-Momentum System" begin

    PARTICLE1, PARTICLE2 = Tuple(rand(TestImplementation.PARTICLE_SET, 2))
    TESTPROC = TestImplementation.TestProcess((PARTICLE1, PARTICLE2), OUTPARTICLE)


    SQRT_Ss = (mass(PARTICLE1) + mass(PARTICLE2)) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))

    ENERGIES1 = mass(PARTICLE1) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))
    ENERGIES2 = mass(PARTICLE2) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))

    @testset "$AXIS" for AXIS in TEST_AXES

        @testset "cms energy" begin
            PSL = CenterOfMomentumSystem(CMSEnergy(), AXIS)
            coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)

            @testset "sqrt_s = $sqrt_s" for sqrt_s in SQRT_Ss

                MOM1, MOM2 = @inferred coord_map((sqrt_s,))

                @test isapprox(getMass(MOM1), mass(PARTICLE1))
                @test isapprox(getMass(MOM2), mass(PARTICLE2))
                @test isapprox(getMass(MOM1 + MOM2), sqrt_s)
                @test isapprox(MOM1[2:4], -MOM2[2:4])
            end
        end

        @testset "energy_1" begin
            PSL = CenterOfMomentumSystem(Energy(1), AXIS)
            coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)

            @testset "E1= $E1" for E1 in ENERGIES1

                MOM1, MOM2 = @inferred coord_map((E1,))

                @test isapprox(getMass(MOM1), mass(PARTICLE1))
                @test isapprox(getMass(MOM2), mass(PARTICLE2))
                @test isapprox(getE(MOM1), E1)
                @test isapprox(MOM1[2:4], -MOM2[2:4])
            end
        end

        @testset "energy_2" begin
            PSL = CenterOfMomentumSystem(Energy(2), AXIS)
            coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)

            @testset "E2= $E2" for E2 in ENERGIES2

                MOM1, MOM2 = @inferred coord_map((E2,))

                @test isapprox(getMass(MOM1), mass(PARTICLE1))
                @test isapprox(getMass(MOM2), mass(PARTICLE2))
                @test isapprox(getE(MOM2), E2)
                @test isapprox(MOM1[2:4], -MOM2[2:4])
            end
        end
    end
end
