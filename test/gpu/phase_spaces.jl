using QEDcore
using Random

RNG = Random.MersenneTwister(573)

N = 256

TESTMODEL = TestImplementation.TestModel()

@testset "phase space tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
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
        proc = TestImplementation.TestProcess((Electron(), Photon()), (Electron(), Photon()))
        in_psl = TestImplementation.TestInPhaseSpaceLayout()
        psl = FlatPhaseSpaceLayout(in_psl)

        psps = PhaseSpacePoint.(proc, model, Ref(psl), tuple.(in_els, in_phs), tuple.(out_els, out_phs))
        gpu_psps = PhaseSpacePoint.(proc, model, Ref(psl), tuple.(gpu_in_els, gpu_in_phs), tuple.(gpu_out_els, gpu_out_phs))

        @testset "accessors" begin
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Electron())), in_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Photon())), in_ph_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Electron())), out_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Photon())), out_ph_moms)) == N

            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Val(1))), in_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Val(2))), in_ph_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Val(1))), out_el_moms)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Val(2))), out_ph_moms)) == N
        end

        @testset "generation from coordinates" begin
            IN_MASSES = mass.(incoming_particles(proc))
            SUM_IN_MASSES = sum(IN_MASSES)
            OUT_MASSES = mass.(outgoing_particles(proc))
            SUM_OUT_MASSES = sum(OUT_MASSES)
            TESTSQRTS = [(one(FLOAT_T) + rand(RNG, FLOAT_T)) * (SUM_OUT_MASSES + SUM_IN_MASSES) for _ in 1:N]
            TESTINMOMS = TestImplementation._generate_onshell_two_body_moms.(
                Ref(RNG), Ref(IN_MASSES), TESTSQRTS
            )
            test_out_psl = FlatPhaseSpaceLayout(in_psl)
            TESTOUTCOORDS = [Tuple(rand(RNG, 4 * number_outgoing_particles(proc))) for _ in 1:N]

            test_out_moms = QEDbase.build_momenta.(
                proc, model, TESTINMOMS, Ref(test_out_psl), TESTOUTCOORDS
            )

            test_out_moms_gpu = QEDbase.build_momenta.(
                proc, model, VECTOR_T(TESTINMOMS), Ref(test_out_psl), VECTOR_T(TESTOUTCOORDS)
            )

            @test sum(isapprox.(Vector(gpu_test_out_moms), test_out_moms)) == N
        end
    end
end
