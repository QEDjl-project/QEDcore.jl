using QEDcore
using StaticArrays
using Random

RNG = MersenneTwister(2147483647)

unary_methods = [-, +]
binary_array_methods = [+, -]
binary_float_methods = [*, /]

groundtruth_mul(a::AdjointBiSpinor, b::BiSpinor) = transpose(SArray(a)) * SArray(b)
function groundtruth_mul(a::BiSpinor, b::AdjointBiSpinor)
    return DiracMatrix(SArray(a) * transpose(SArray(b)))
end
function groundtruth_mul(a::AdjointBiSpinor, b::DiracMatrix)
    return AdjointBiSpinor(transpose(SArray(a)) * SArray(b))
end
groundtruth_mul(a::DiracMatrix, b::BiSpinor) = BiSpinor(SArray(a) * SArray(b))
groundtruth_mul(a::DiracMatrix, b::DiracMatrix) = DiracMatrix(SArray(a) * SArray(b))

@testset "DiracTensor{$T}" for T in [
        Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64,
    ]

    allowed_muls = Dict(
        [
            (AdjointBiSpinor{T}, BiSpinor{T}) => T,
            (BiSpinor{T}, AdjointBiSpinor{T}) => DiracMatrix{T},
            (AdjointBiSpinor{T}, DiracMatrix{T}) => AdjointBiSpinor{T},
            (DiracMatrix{T}, BiSpinor{T}) => BiSpinor{T},
            (DiracMatrix{T}, DiracMatrix{T}) => DiracMatrix{T},
        ]
    )

    dirac_tensors = [
        BiSpinor(rand(RNG, T, 4)),
        AdjointBiSpinor(rand(RNG, T, 4)),
        DiracMatrix(rand(RNG, T, 4, 4)),
    ]

    @testset "BiSpinor" begin
        BS = BiSpinor(rand(RNG, T, 4))

        @test size(BS) == (4,)
        @test length(BS) == 4
        @test eltype(BS) == T

        @test @inferred(BiSpinor{T}(1, 2, 3, 4)) == @inferred(BiSpinor{T}([1, 2, 3, 4]))

        BS1 = BiSpinor{T}(1, 2, 3, 4)
        BS2 = BiSpinor{T}(4, 3, 2, 1)

        @test @inferred(BS1 + BS2) == BiSpinor{T}(5, 5, 5, 5)
        @test @inferred(BS1 - BS2) == BiSpinor{T}(-3, -1, 1, 3)

        @test_throws DimensionMismatch(
            "No precise constructor for BiSpinor{$T} found. Length of input was 2."
        ) BiSpinor{T}(1, 2)
    end #BiSpinor

    @testset "AdjointBiSpinor" begin
        aBS = AdjointBiSpinor(rand(RNG, T, 4))

        @test size(aBS) == (4,)
        @test length(aBS) == 4
        @test eltype(aBS) == T

        @test @inferred(AdjointBiSpinor{T}(1, 2, 3, 4)) ==
            @inferred(AdjointBiSpinor{T}([1, 2, 3, 4]))

        aBS1 = AdjointBiSpinor{T}(1, 2, 3, 4)
        aBS2 = AdjointBiSpinor{T}(4, 3, 2, 1)

        @test @inferred(aBS1 + aBS2) == AdjointBiSpinor{T}(5, 5, 5, 5)
        @test @inferred(aBS1 - aBS2) == AdjointBiSpinor{T}(-3, -1, 1, 3)

        @test_throws DimensionMismatch(
            "No precise constructor for AdjointBiSpinor{$T} found. Length of input was 2."
        ) AdjointBiSpinor{T}(1, 2)
    end #AdjointBiSpinor

    @testset "DiracMatrix" begin
        DM = DiracMatrix(rand(RNG, T, (4, 4)))

        @test size(DM) == (4, 4)
        @test length(DM) == 16
        @test eltype(DM) == T

        DM1 = DiracMatrix{T}(SDiagonal(1, 2, 3, 4))
        DM2 = DiracMatrix{T}(SDiagonal(4, 3, 2, 1))

        @test @inferred(DM1 + DM2) == DiracMatrix{T}(SDiagonal(5, 5, 5, 5))
        @test @inferred(DM1 - DM2) == DiracMatrix{T}(SDiagonal(-3, -1, 1, 3))

        @test_throws DimensionMismatch(
            "No precise constructor for DiracMatrix{$T} found. Length of input was 2."
        ) DiracMatrix{T}(1, 2)
    end #DiracMatrix

    @testset "General Arithmetics" begin
        num = rand(RNG, T)

        for ten in dirac_tensors
            @testset "$ops($(typeof(ten)))" for ops in unary_methods
                res = ops(ten)
                @test typeof(res) == typeof(ten)
                @test SArray(res) == ops(SArray(ten))
            end

            @testset "num*$(typeof(ten))" begin
                @test @inferred(ten * num) == @inferred(num * ten)
                @test SArray(num * ten) == num * SArray(ten)
                @test typeof(num * ten) == typeof(ten)
            end

            @testset "$(typeof(ten))/num" begin
                res_float_div = ten / num
                @test SArray(@inferred(ten / num)) == SArray(ten) / num
                @test typeof(ten / num) == typeof(ten)
            end

            @testset "$(typeof(ten))*$(typeof(ten2))" for ten2 in dirac_tensors
                mul_comb = (typeof(ten), typeof(ten2))
                if mul_comb in keys(allowed_muls)
                    res = ten * ten2
                    @test typeof(res) == allowed_muls[mul_comb]
                    @test isapprox(res, groundtruth_mul(ten, ten2))
                else
                    @test_throws MethodError ten * ten2
                end
            end
        end
    end #Arithmetics
end #"DiracTensor"


allowed_muls = Dict(
    [
        (AdjointBiSpinor, BiSpinor) => ComplexF64,
        (BiSpinor, AdjointBiSpinor) => DiracMatrix,
        (AdjointBiSpinor, DiracMatrix) => AdjointBiSpinor,
        (DiracMatrix, BiSpinor) => BiSpinor,
        (DiracMatrix, DiracMatrix) => DiracMatrix,
    ]
)

@testset "promotion (multiplication $T1 * $T2)" for (T1, T2) in keys(allowed_muls)
    a1 = rand(RNG, T1{Float32})
    a2 = rand(RNG, T2{ComplexF64})

    RES_T = allowed_muls[(T1, T2)]
    if RES_T != ComplexF64
        RES_T = RES_T{ComplexF64}
    end
    @test a1 * a2 isa RES_T
    @test isapprox(a1 * a2, groundtruth_mul(a1, a2))
end
