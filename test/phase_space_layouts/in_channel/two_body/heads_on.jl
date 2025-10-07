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

    @show mass(PARTICLE1)
    @show mass(PARTICLE2)

    TESTPROC = TestImplementation.TestProcess((PARTICLE1, PARTICLE2), OUTPARTICLE)

    ENERGIES1 = mass(PARTICLE1) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))
    ENERGIES2 = mass(PARTICLE2) .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))

    @testset "$AXIS" for AXIS in TEST_AXES
        PSL = HeadsOnSystem(AXIS, PARTICLE1, PARTICLE2)
        coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)
        @testset "E1 = $E1, E2 = $E2" for (E1, E2) in Iterators.product(ENERGIES1, ENERGIES2)
            MOM1, MOM2 = coord_map((E1, E2))

            rho1 = _rho_from_E(E1, mass(PARTICLE1))
            rho2 = _rho_from_E(E2, mass(PARTICLE2))

            @test isapprox(getMass(MOM1), mass(PARTICLE1))
            @test isapprox(getMass(MOM2), mass(PARTICLE2))
            @test isapprox(getE(MOM1), E1)
            @test isapprox(getE(MOM2), E2)
            @test isapprox(MOM1[2:4] ./ rho1, MOM2[2:4] ./ rho2)

        end
    end
end

@testset "Photon-Electron" begin
    INPARTICLES = (Electron(), Photon())
    TESTPROC = TestImplementation.TestProcess(INPARTICLES, OUTPARTICLE)
    ENERGIES = 1.0 .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))
    OMEGAS = (1.0e-6 * rand(RNG), 1.0e3 * rand(RNG), rand(RNG), 1.0e3 * rand(RNG), 1.0e6 * rand(RNG))
    @testset "$AXIS" for AXIS in TEST_AXES
        PSL = PhotonElectronHeadsOnSystem(AXIS)
        coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)
        @testset "E = $E, omega = $omega" for (E, omega) in Iterators.product(ENERGIES, OMEGAS)


            P, K = coord_map((E, omega))
            rho = _rho_from_E(E, mass(Electron()))

            @test isapprox(getMass(P), 1.0)
            @test isapprox(getMass(K), 0.0)
            @test isapprox(getE(P), E)
            @test isapprox(getE(K), omega)
            @test isapprox(K[2:4] / omega, -P[2:4] / rho)

        end
    end
end
