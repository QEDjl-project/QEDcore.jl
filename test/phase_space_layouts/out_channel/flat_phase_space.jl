using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestPerturbativeModel()
TESTPSDEF = TestImplementation.TestPhasespaceDef()
TESTINPSL = TestImplementation.TestInPhaseSpaceLayout()
