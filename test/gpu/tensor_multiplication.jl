if isempty(GPUS)
    @info """No GPU tests are enabled, skipping tests...
    To test GPU functionality, please use 'TEST_<GPU> = 1 julia ...' for one of GPU=[CUDA, AMDGPU, METAL, ONEAPI]"""
    return nothing
end

using QEDcore
using Random

RNG = Random.MersenneTwister(573)

ALLOWED_MULS = [
    (AdjointBiSpinor, BiSpinor, Union),
    (BiSpinor, AdjointBiSpinor, DiracMatrix),
    (AdjointBiSpinor, DiracMatrix, AdjointBiSpinor),
    (DiracMatrix, BiSpinor, BiSpinor),
    (DiracMatrix, DiracMatrix, DiracMatrix),
]


const N = 256

@testset "gpu tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        @testset "multiplication $T1 * $T2 = $T3" for (T1, T2, T3) in ALLOWED_MULS
            COMPLEX_T = Complex{FLOAT_T}

            par1 = rand(RNG, T1{COMPLEX_T}, N)
            par2 = rand(RNG, T2{COMPLEX_T}, N)

            gt_results = par1 .* par2

            par1_gpu = VECTOR_T(par1)
            par2_gpu = VECTOR_T(par2)

            gpu_results = par1_gpu .* par2_gpu

            @test all(isapprox.(Vector(gpu_results), gt_results))
        end
    end
end
