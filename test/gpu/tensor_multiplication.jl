if isempty(GPUS)
    @info """No GPU tests are enabled, skipping tests...
    To test GPU functionality, please use 'TEST_<GPU> = 1 julia ...' for one of GPU=[CUDA, AMDGPU, METAL, ONEAPI]"""
    return nothing
end

using QEDcore
using Random

RNG = Random.MersenneTwister(573)

ALLOWED_MULS = [
    (BiSpinor, Union, BiSpinor),
    (AdjointBiSpinor, Union, AdjointBiSpinor),
    (DiracMatrix, Union, DiracMatrix),
    (AdjointBiSpinor, BiSpinor, Union),
    (BiSpinor, AdjointBiSpinor, DiracMatrix),
    (AdjointBiSpinor, DiracMatrix, AdjointBiSpinor),
    (DiracMatrix, BiSpinor, BiSpinor),
    (DiracMatrix, DiracMatrix, DiracMatrix),
]

const N = 256

@testset "gpu tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        COMPLEX_T = Complex{FLOAT_T}

        @testset "multiplication $T1 * $T2 = $T3" for (T1, T2, T3) in ALLOWED_MULS
            par1 = rand(RNG, T1{COMPLEX_T}, N)
            par2 = rand(RNG, T2{COMPLEX_T}, N)

            gt_results = par1 .* par2

            par1_gpu = VECTOR_T(par1)
            par2_gpu = VECTOR_T(par2)

            gpu_results = par1_gpu .* par2_gpu

            @test eltype(gpu_results) == T3{COMPLEX_T}
            @test all(isapprox.(Vector(gpu_results), gt_results))
        end

        @testset "multiplication AdjointBiSpinor * DiracMatrix * BiSpinor" begin
            par1 = rand(RNG, AdjointBiSpinor{COMPLEX_T}, N)
            par2 = rand(RNG, DiracMatrix{COMPLEX_T}, N)
            par3 = rand(RNG, BiSpinor{COMPLEX_T}, N)

            gt_results = par1 .* par2 .* par3

            par1_gpu = VECTOR_T(par1)
            par2_gpu = VECTOR_T(par2)
            par3_gpu = VECTOR_T(par3)

            gpu_results = par1_gpu .* par2_gpu .* par3_gpu

            @test eltype(gpu_results) == COMPLEX_T
            @test all(isapprox.(Vector(gpu_results), gt_results))
        end
    end
end
