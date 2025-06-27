using QEDcore

CONSTANTS = (
    ALPHA,
    ALPHA_SQUARE,
    ELEMENTARY_CHARGE,
    ELEMENTARY_CHARGE_SQUARE,
    ELECTRONMASS,
    ONE_OVER_FOURPI,
)

TYPES = (Float16, Float32, Float64)

@testset "type check" begin
    @testset "$C" for C in CONSTANTS

        def_val = big(C)
        @testset "$T" for T in TYPES
            typed_val = T(def_val)
            @test T(C) == typed_val
            @test one(T) * C == typed_val
            @test T(T(C)) = T(C)
        end
    end
end

@testset "sanity checks" begin
    @test isapprox(ELEMENTARY_CHARGE_SQUARE, ELEMENTARY_CHARGE^2)
    @test isapprox(ALPHA_SQUARE, ALPHA^2)
    @test isapprox(ONE_OVER_FOURPI, inv(4 * pi))
    @test isapprox(ELEMENTARY_CHARGE_SQUARE * ONE_OVER_FOURPI, ALPHA)
end
