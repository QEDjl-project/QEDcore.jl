using QEDcore
using Random

RNG = Random.MersenneTwister(573)

N = 256

@testset "base state and propagator tests for $GPU_MODULE" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        COMPLEX_T = Complex{FLOAT_T}
        MOMENTUM_TYPE = SFourMomentum{FLOAT_T}

        moms = rand(RNG, MOMENTUM_TYPE, N)
        gpu_moms = VECTOR_T(moms)

        @testset "base state for $dir $particle with $sp" for (dir, particle, sp) in Iterators.product(
                (Incoming(), Outgoing()),
                (Electron(), Positron(), Photon()),
                (SpinUp(), SpinDown(), AllSpin(), PolX(), PolY(), AllPol())
            )
            if is_fermion(particle) && sp isa AbstractPolarization
                continue
            elseif is_boson(particle) && sp isa AbstractSpin
                continue
            end

            base_states = base_state.(particle, dir, moms, sp)
            gpu_base_states = base_state.(particle, dir, gpu_moms, sp)

            EXPECTED_T = is_boson(particle) ? FLOAT_T : COMPLEX_T
            if sp isa Union{AbstractDefiniteSpin, AbstractDefinitePolarization}
                @test eltype(eltype(gpu_base_states)) == EXPECTED_T
            else
                @test eltype(eltype(eltype(gpu_base_states))) == EXPECTED_T
            end

            @test sum(isapprox.(base_states, Vector(gpu_base_states))) == N
        end
    end
end
