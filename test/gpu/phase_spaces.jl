using QEDcore
using Random

RNG = Random.MersenneTwister(573)

N = 256

TESTMODEL = TestImplementation.TestPerturbativeModel()

@testset "phase space tests for $VECTOR_T" for (GPU_MODULE, VECTOR_T) in GPUS
    @testset "float type $FLOAT_T" for FLOAT_T in GPU_FLOAT_TYPES[GPU_MODULE]
        MOMENTUM_TYPE = SFourMomentum{FLOAT_T}

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
        in_psl = TwoBodyTargetSystem()
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
            IN_MASSES = mass.(FLOAT_T, incoming_particles(proc))
            SUM_IN_MASSES = sum(IN_MASSES)
            OUT_MASSES = mass.(FLOAT_T, outgoing_particles(proc))
            SUM_OUT_MASSES = sum(OUT_MASSES)
            SQRTS = [(one(FLOAT_T) + rand(RNG, FLOAT_T)) * (SUM_OUT_MASSES + SUM_IN_MASSES) for _ in 1:N]
            IN_MOMS = TestImplementation._generate_onshell_two_body_moms.(
                Ref(RNG), Ref(IN_MASSES), SQRTS
            )
            out_psl = FlatPhaseSpaceLayout(in_psl)
            OUT_COORDS = [Tuple(rand(RNG, FLOAT_T, 4 * number_outgoing_particles(proc))) for _ in 1:N]

            out_moms = QEDbase.build_momenta.(
                proc, model, IN_MOMS, out_psl, OUT_COORDS
            )

            gpu_out_moms = QEDbase.build_momenta.(
                proc, model, VECTOR_T(IN_MOMS), out_psl, VECTOR_T(OUT_COORDS)
            )

            RTOL = sqrt(eps(FLOAT_T))

            @test eltype(eltype(gpu_out_moms)) == MOMENTUM_TYPE

            @test sum(isapprox.(Vector(getindex.(gpu_out_moms, 1)), getindex.(out_moms, 1); rtol = RTOL)) == N
            @test sum(isapprox.(Vector(getindex.(gpu_out_moms, 2)), getindex.(out_moms, 2); rtol = RTOL)) == N

            IN_COORDS = tuple.(rand(RNG, FLOAT_T, N))

            psps = PhaseSpacePoint.(proc, model, out_psl, IN_COORDS, OUT_COORDS)

            gpu_psps = PhaseSpacePoint.(proc, model, out_psl, VECTOR_T(IN_COORDS), VECTOR_T(OUT_COORDS))

            @test momentum_eltype(eltype(gpu_psps)) == FLOAT_T

            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Val(1))), momentum.(psps, Incoming(), Val(1)); rtol = RTOL)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Incoming(), Val(2))), momentum.(psps, Incoming(), Val(2)); rtol = RTOL)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Val(1))), momentum.(psps, Outgoing(), Val(1)); rtol = RTOL)) == N
            @test sum(isapprox.(Vector(momentum.(gpu_psps, Outgoing(), Val(2))), momentum.(psps, Outgoing(), Val(2)); rtol = RTOL)) == N
        end
    end
end
