using Random
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

include("../../../test_implementation/TestImplementation.jl")

TESTMODEL = TestImplementation.TestPerturbativeModel()

N_OUTGOING = 2
OUTPARTICLE = Tuple(rand(TestImplementation.PARTICLE_SET, N_OUTGOING))

UNI_RUN_COORDS = (Energy, SpatialMagnitude, Rapidity)

@testset "$PART1 $PART2" for (PART1, PART2) in Iterators.product(
        TestImplementation.PARTICLE_SET, TestImplementation.PARTICLE_SET
    )
    MASSES = mass.((PART1, PART2))

    TESTPROC = TestImplementation.TestProcess((PART1, PART2), OUTPARTICLE)

    @testset "run=$RUNIDX rest=$RESTIDX" for (RUNIDX, RESTIDX) in ((1, 2), (2, 1))
        mass_run = MASSES[RUNIDX]
        mass_rest = MASSES[RESTIDX]

        @testset "$COORD" for COORD in UNI_RUN_COORDS
            test_psl_default = TwoBodyRestSystem{RESTIDX}(COORD(RUNIDX))

            @testset "constructors" begin
                test_psl_call = TwoBodyRestSystem(RESTIDX, COORD(RUNIDX))
                test_psl_val = TwoBodyRestSystem(Val(RESTIDX), COORD(RUNIDX))
                test_psl_infered = TwoBodyRestSystem(COORD(RUNIDX))

                @test test_psl_default === test_psl_call
                @test test_psl_default === test_psl_val
                @test test_psl_default === test_psl_infered
            end

            @testset "Error Handling" begin
                @test_throws InvalidInputError TwoBodyRestSystem(RESTIDX, COORD(RESTIDX))
            end
        end

        @testset "Energy" begin
            test_psl_energy = TwoBodyRestSystem(Energy(RUNIDX))
            coord_map_energy = CoordinateMap(TESTPROC, TESTMODEL, test_psl_energy)

            @testset "en = $en" for en in (mass_run, mass_run + rand(RNG))
                in_moms = coord_map_energy((en,))
                in_mom_run = in_moms[RUNIDX]
                in_mom_rest = in_moms[RESTIDX]

                @test isapprox(getMass(in_mom_run), mass_run, atol = ATOL, rtol = RTOL)
                @test isapprox(getMass(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
                @test isapprox(getE(in_mom_run), en, atol = ATOL, rtol = RTOL)
                @test isapprox(getE(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
            end
        end

        @testset "SpatialMagnitude" begin
            test_psl_mag = TwoBodyRestSystem(SpatialMagnitude(RUNIDX))
            coord_map_mag = CoordinateMap(TESTPROC, TESTMODEL, test_psl_mag)

            @testset "rho = $rho" for rho in (0.0, rand(RNG))
                in_moms = coord_map_mag((rho,))
                in_mom_run = in_moms[RUNIDX]
                in_mom_rest = in_moms[RESTIDX]

                @test isapprox(getMass(in_mom_run), mass_run, atol = ATOL, rtol = RTOL)
                @test isapprox(getMass(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
                @test isapprox(getMag(in_mom_run), rho, atol = ATOL, rtol = RTOL)
                @test isapprox(getE(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
            end
        end

        @testset "Rapidity" begin
            test_psl_rap = TwoBodyRestSystem(Rapidity(RUNIDX))
            coord_map_rap = CoordinateMap(TESTPROC, TESTMODEL, test_psl_rap)

            @testset "y= $y" for y in (0.0, rand(RNG))
                in_moms = coord_map_rap((y,))
                in_mom_run = in_moms[RUNIDX]
                in_mom_rest = in_moms[RESTIDX]

                @test isapprox(getMass(in_mom_run), mass_run, atol = ATOL, rtol = RTOL)
                @test isapprox(getMass(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
                @test isapprox(getRapidity(in_mom_run), y, atol = ATOL, rtol = RTOL)
                @test isapprox(getE(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
            end
        end

        @testset "center-of-momentum system" begin
            test_psl_default = TwoBodyRestSystem{RESTIDX}(CMSEnergy())

            @testset "constructors" begin
                test_psl_call = TwoBodyRestSystem(RESTIDX, CMSEnergy())
                test_psl_val = TwoBodyRestSystem(Val(RESTIDX), CMSEnergy())

                @test test_psl_default === test_psl_call
                @test test_psl_default === test_psl_val
            end

            @testset "build momenta" begin
                test_psl_cms = TwoBodyRestSystem(RESTIDX, CMSEnergy())
                coord_map_cms = CoordinateMap(TESTPROC, TESTMODEL, test_psl_cms)
                mass_sum = sum(MASSES)

                @testset "ss = $ss" for ss in (mass_sum, mass_sum + rand(RNG))
                    in_moms = coord_map_cms((ss,))
                    in_mom_run = in_moms[RUNIDX]
                    in_mom_rest = in_moms[RESTIDX]
                    in_mom_sum = sum(in_moms)

                    @test isapprox(getMass(in_mom_run), mass_run, atol = ATOL, rtol = RTOL)
                    @test isapprox(getMass(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
                    @test isapprox(sqrt(in_mom_sum * in_mom_sum), ss, atol = ATOL, rtol = RTOL)
                    @test isapprox(getE(in_mom_rest), mass_rest, atol = ATOL, rtol = RTOL)
                end
            end
        end
    end
end
