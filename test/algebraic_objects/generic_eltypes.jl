using QEDcore
using Random

RNG = MersenneTwister(0)

@testset "generic eltypes for $CONTAINER_T" for CONTAINER_T in [
        DiracMatrix,
        BiSpinor,
        AdjointBiSpinor,
        SFourMomentum,
        SLorentzVector,
        MLorentzVector,
    ]
    @testset "eltype $EL_T" for EL_T in [Float16, Float32, Float64, Int16, Int32, Int64]
        a1 = zero(CONTAINER_T{EL_T})
        a2 = rand(RNG, CONTAINER_T{EL_T})
        @test isapprox(a1, zero(CONTAINER_T))
        @test zero(CONTAINER_T{EL_T}) isa CONTAINER_T{EL_T}
        @test a2 isa CONTAINER_T{EL_T}
        @test (a1 + a2) isa CONTAINER_T{EL_T}
        @test isapprox((a1 + a2), a2)
    end

    @testset "type promotions (addition)" begin
        T1 = Float32
        T2 = if CONTAINER_T == SFourMomentum
            Float64
        else
            ComplexF64
        end
        a1 = rand(RNG, CONTAINER_T{T1})
        a2 = rand(RNG, CONTAINER_T{T2})

        @test a1 + a2 isa CONTAINER_T{T2}
        @test all([isapprox(x + y, z) for (x, y, z) in zip(a1, a2, a1 + a2)])
    end
end
