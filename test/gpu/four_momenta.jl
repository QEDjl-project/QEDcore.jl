using QEDcore
using Random

RNG = Random.MersenneTwister(573)

N = 256

@testset "four momentum tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        RTOL = 4 * eps(FLOAT_T)
        ATOL = 4 * eps(FLOAT_T)

        MOMENTUM_TYPE = SFourMomentum{FLOAT_T}

        xs = rand(RNG, FLOAT_T, N)
        ys = rand(RNG, FLOAT_T, N)
        zs = rand(RNG, FLOAT_T, N)
        masses = rand(RNG, FLOAT_T, N)
        Es = @. sqrt(xs^2 + ys^2 + zs^2 + masses^2)

        moms = MOMENTUM_TYPE.(Es, xs, ys, zs)

        gpu_masses = VECTOR_T(masses)
        gpu_moms = VECTOR_T(moms)

        @testset "magnitudes and masses" begin
            @test sum(isapprox.(Vector(getMag2.(gpu_moms)), getMag2.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMag.(gpu_moms)), getMag.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMass2.(gpu_moms)), getMass2.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMass.(gpu_moms)), getMass.(moms); atol = ATOL, rtol = RTOL)) == N
        end

        @testset "components" begin
            @test sum(isapprox.(Vector(getE.(gpu_moms)), getE.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getPx.(gpu_moms)), getPx.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getPy.(gpu_moms)), getPy.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getPz.(gpu_moms)), getPz.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getBeta.(gpu_moms)), getBeta.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getGamma.(gpu_moms)), getGamma.(moms); atol = ATOL, rtol = RTOL)) == N
        end

        @testset "transverse coordinates" begin
            @test sum(isapprox.(Vector(getPt2.(gpu_moms)), getPt2.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getPt.(gpu_moms)), getPt.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMt2.(gpu_moms)), getMt2.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMt.(gpu_moms)), getMt.(moms); atol = ATOL, rtol = RTOL)) == N
        end

        @testset "spherical coordinates" begin
            @test sum(isapprox.(Vector(getRho.(gpu_moms)), getRho.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getCosTheta.(gpu_moms)), getCosTheta.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getCosPhi.(gpu_moms)), getCosPhi.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getSinPhi.(gpu_moms)), getSinPhi.(moms); atol = ATOL, rtol = RTOL)) == N
        end

        @testset "light-cone coordinates" begin
            @test sum(isapprox.(Vector(getPlus.(gpu_moms)), getPlus.(moms); atol = ATOL, rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getMinus.(gpu_moms)), getMinus.(moms); atol = ATOL, rtol = RTOL)) == N
        end

        @testset "isonshell" begin
            @test sum(Vector(isonshell.(gpu_moms, gpu_masses))) == N
        end
    end
end
