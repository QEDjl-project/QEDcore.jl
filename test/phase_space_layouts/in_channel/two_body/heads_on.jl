using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestPerturbativeModel()

INPARTICLES = (Electron(), Photon())

N_OUTGOING = 2
OUTPARTICLE = Tuple(rand(TestImplementation.PARTICLE_SET, N_OUTGOING))

TEST_AXES = (XAxis(), YAxis(), ZAxis())

TESTPROC = TestImplementation.TestProcess(INPARTICLES, OUTPARTICLE)

ENERGIES = 1.0 .+ (rand(RNG), 10 * rand(RNG), 100 * rand(RNG))
OMEGAS = (1.0e-6 * rand(RNG), 1.0e3 * rand(RNG), rand(RNG), 1.0e3 * rand(RNG), 1.0e6 * rand(RNG))


@testset "$AXIS" for AXIS in TEST_AXES
    PSL = PhotonElectronHeadsOnSystem(AXIS)
    coord_map = CoordinateMap(TESTPROC, TESTMODEL, PSL)
    @testset "E = $E, omega = $omega" for (E, omega) in Iterators.product(ENERGIES, OMEGAS)


        P, K = coord_map((E, omega))

        @test isapprox(getMass(P), 1.0)
        @test isapprox(getMass(K), 0.0)
        @test isapprox(getE(P), E)
        @test isapprox(getE(K), omega)

    end
end
