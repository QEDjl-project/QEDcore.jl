using Random
using StatsBase
using QEDcore

RNG = MersenneTwister(137137)
ATOL = 0.0
RTOL = sqrt(eps())

RNDIDX = rand(RNG, 1:8)
COORD_BASE_NAMES = Dict(
    Energy => "energy",
    CosTheta => "cos_theta",
    SpatialMagnitude => "spatial_magnitude",
    Rapidity => "rapidity",
)

UNI_COORDS = (Energy, CosTheta, SpatialMagnitude, Rapidity)
@testset "univariate" begin
    @testset "$COORD" for COORD in UNI_COORDS
        test_coord = COORD(RNDIDX)

        @test test_coord isa COORD{RNDIDX}
        @test particle_index(test_coord) == RNDIDX
        @test coordinate_name(test_coord) == COORD_BASE_NAMES[COORD] * "_$RNDIDX"
        @test coordinate_names(test_coord) == (COORD_BASE_NAMES[COORD] * "_$RNDIDX",)

        test_coord_val = COORD(Val(RNDIDX))
        test_coord_direct = COORD{RNDIDX}()
        @test test_coord_val === test_coord
        @test test_coord_direct === test_coord
    end
end

@testset "multivariate" begin
    @testset "$DIM" for DIM in (1, 2, rand(RNG, 3:8))
        rnd_indices = sample(RNG, 1:20, DIM; replace=false)
        MULTI_COORDS = Tuple(rand(RNG, UNI_COORDS, DIM))

        COORDS = Tuple(
            collect(coord(idx) for (coord, idx) in zip(MULTI_COORDS, rnd_indices))
        )
        COORD_NAMES = Tuple(
            collect(
                "$(COORD_BASE_NAMES[coord])" * "_$idx" for
                (coord, idx) in zip(MULTI_COORDS, rnd_indices)
            ),
        )

        test_coord_set = CoordinateSet(COORDS)

        @testset "construction" begin
            test_coord_set_unroll = CoordinateSet(COORDS...)
            test_coord_set_direct = CoordinateSet{DIM}(COORDS)
            test_coord_set_direct_unroll = CoordinateSet{DIM}(COORDS...)

            @test test_coord_set == test_coord_set_unroll
            @test test_coord_set == test_coord_set_direct
            @test test_coord_set == test_coord_set_direct_unroll
        end

        @testset "utility" begin
            @test coordinate_names(test_coord_set) == COORD_NAMES
            @test phase_space_dimension(test_coord_set) == DIM
        end

        @testset "error handling" begin
            MULTI_COORDS_TOO_MANY = Tuple(rand(RNG, UNI_COORDS, DIM + 1))
            MULTI_COORDS_FEWER = Tuple(rand(RNG, UNI_COORDS, DIM - 1))
            rnd_indices_too_many = sample(RNG, 1:20, DIM + 1; replace=false)
            rnd_indices_fewer = sample(RNG, 1:20, DIM - 1; replace=false)
            COORDS_TO_MANY = Tuple(
                collect(
                    coord(idx) for
                    (coord, idx) in zip(MULTI_COORDS_TOO_MANY, rnd_indices_too_many)
                ),
            )
            COORDS_FEWER = Tuple(
                collect(
                    coord(idx) for
                    (coord, idx) in zip(MULTI_COORDS_FEWER, rnd_indices_fewer)
                ),
            )

            @test_throws ArgumentError CoordinateSet{DIM}(COORDS_TO_MANY)
            @test_throws ArgumentError CoordinateSet{DIM}(COORDS_TO_MANY...)
            @test_throws ArgumentError CoordinateSet{DIM}(COORDS_FEWER)
            @test_throws ArgumentError CoordinateSet{DIM}(COORDS_FEWER...)
        end
    end
end
