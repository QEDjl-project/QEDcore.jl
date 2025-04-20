using QEDcore
using Random

RNG = Random.MersenneTwister(573)

N = 256

TESTMODEL = TestImplementation.TestModel()
TESTPSL = TestImplementation.TestOutPhaseSpaceLayout()

@testset "four momentum tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        MOMENTUM_TYPE = SFourMomentum{FLOAT_T}
        COMPLEX_T = Complex{FLOAT_T}

        in_el_moms = rand(RNG, MOMENTUM_TYPE, N)
        in_ph_moms = rand(RNG, MOMENTUM_TYPE, N)
        out_el_moms = rand(RNG, MOMENTUM_TYPE, N)
        out_ph_moms = rand(RNG, MOMENTUM_TYPE, N)

        in_els = ParticleStateful.(Incoming(), Electron(), in_el_moms)
        in_phs = ParticleStateful.(Incoming(), Photon(), in_ph_moms)
        out_els = ParticleStateful.(Outgoing(), Electron(), out_el_moms)
        out_phs = ParticleStateful.(Outgoing(), Photon(), out_ph_moms)

        gpu_in_els = VECTOR_T(in_els)
        gpu_in_phs = VECTOR_T(in_phs)
        gpu_out_els = VECTOR_T(out_els)
        gpu_out_phs = VECTOR_T(out_phs)

        model = TESTMODEL
        process = TestImplementation.TestProcess((Electron(), Photon()), (Electron(), Photon()))
        psl = TESTPSL

        psps = PhaseSpacePoint.(process, model, psl, tuple.(in_els, in_phs), tuple.(out_els, out_phs))
        gpu_psps = PhaseSpacePoint.(process, model, psl, tuple.(gpu_in_els, gpu_in_phs), tuple.(gpu_out_els, gpu_out_phs))

        @testset "Accessors" begin
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Electron())), in_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Photon())), in_ph_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Electron())), out_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Photon())), out_ph_moms)) == N
        end
    end
end
