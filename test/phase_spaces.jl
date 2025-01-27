using Random
using StaticArrays
using QEDcore

include("test_implementation/TestImplementation.jl")
TESTMODEL = TestImplementation.TestModel()
TESTPSL = TestImplementation.TestOutPhaseSpaceLayout()

RNG = Random.MersenneTwister(727)
BUF = IOBuffer()

@testset "broadcast" begin
    test_func(psl) = psl
    @test test_func.(TESTPSL) == TESTPSL
end

@testset "Stateful Particle" begin
    DIRECTIONS = [Incoming(), Outgoing()]
    SPECIES = [Electron(), Positron()] #=, Muon(), AntiMuon(), Tauon(), AntiTauon()=#

    for (species, dir) in Iterators.product(SPECIES, DIRECTIONS)
        mom = rand(RNG, SFourMomentum)

        particle_stateful = ParticleStateful(dir, species, mom)
        @test particle_stateful == ParticleStateful{typeof(dir),typeof(species)}(mom)
        @test particle_stateful ==
            ParticleStateful{typeof(dir),typeof(species),typeof(mom)}(mom)

        # particle interface
        @test is_fermion(particle_stateful) == is_fermion(species)
        @test is_boson(particle_stateful) == is_boson(species)
        @test is_particle(particle_stateful) == is_particle(species)
        @test is_anti_particle(particle_stateful) == is_anti_particle(species)
        @test is_incoming(particle_stateful) == is_incoming(dir)
        @test is_outgoing(particle_stateful) == is_outgoing(dir)
        @test mass(particle_stateful) == mass(species)
        @test charge(particle_stateful) == charge(species)

        # accessors
        @test particle_stateful.dir == dir
        @test particle_direction(particle_stateful) == particle_stateful.dir
        @test particle_stateful.species == species
        @test particle_species(particle_stateful) == particle_stateful.species
        @test particle_stateful.mom == mom
        @test momentum(particle_stateful) == mom

        # printing
        print(BUF, particle_stateful)
        @test String(take!(BUF)) == "$(dir) $(species): $(mom)"

        show(BUF, MIME"text/plain"(), particle_stateful)
        @test String(take!(BUF)) ==
            "ParticleStateful: $(dir) $(species)\n    momentum: $(mom)\n"
    end
end

@testset "Phasespace Point" begin
    in_el_mom = rand(RNG, SFourMomentum)
    in_ph_mom = rand(RNG, SFourMomentum)
    out_el_mom = rand(RNG, SFourMomentum)
    out_ph_mom = rand(RNG, SFourMomentum)

    in_el = ParticleStateful(Incoming(), Electron(), in_el_mom)
    in_ph = ParticleStateful(Incoming(), Photon(), in_ph_mom)
    out_el = ParticleStateful(Outgoing(), Electron(), out_el_mom)
    out_ph = ParticleStateful(Outgoing(), Photon(), out_ph_mom)

    in_particles_valid = (in_el, in_ph)
    in_particles_invalid = (in_el, out_ph)

    out_particles_valid = (out_el, out_ph)
    out_particles_invalid = (out_el, in_ph)

    model = TESTMODEL
    process = TestImplementation.TestProcess((Electron(), Photon()), (Electron(), Photon()))
    psl = TESTPSL

    psp = PhaseSpacePoint(process, model, psl, in_particles_valid, out_particles_valid)

    take!(BUF)
    print(BUF, psp)
    @test String(take!(BUF)) == "PhaseSpacePoint of $(process)"

    #=
    show(BUF, MIME"text/plain"(), psp)
    @test_broken match(
        r"PhaseSpacePoint:\n    process: (.*)TestProcess(.*)\n    model: (.*)TestModel(.*)\n    phasespace definition: (.*)TestPhasespaceDef(.*)\n    incoming particles:\n     -> incoming electron: (.*)\n     -> incoming photon: (.*)\n    outgoing particles:\n     -> outgoing electron: (.*)\n     -> outgoing photon: (.*)\n",
        String(take!(BUF)),
    ) isa RegexMatch
    =#

    @testset "Accessor" begin
        @test momentum(psp, Incoming(), 1) == in_el.mom
        @test momentum(psp, Incoming(), 2) == in_ph.mom
        @test momentum(psp, Outgoing(), 1) == out_el.mom
        @test momentum(psp, Outgoing(), 2) == out_ph.mom

        @test psp[Incoming(), 1] == in_el
        @test psp[Incoming(), 2] == in_ph
        @test psp[Outgoing(), 1] == out_el
        @test psp[Outgoing(), 2] == out_ph
    end

    @testset "Error handling" begin
        if (VERSION >= v"1.8")
            # julia versions before 1.8 did not have support for regex matching in @test_throws
            @test_throws r"expected incoming photon but got outgoing photon" PhaseSpacePoint(
                process, model, psl, in_particles_invalid, out_particles_valid
            )

            @test_throws r"expected outgoing photon but got incoming photon" PhaseSpacePoint(
                process, model, psl, in_particles_valid, out_particles_invalid
            )

            @test_throws r"expected incoming electron but got incoming photon" PhaseSpacePoint(
                process, model, psl, (in_ph, in_el), out_particles_valid
            )

            @test_throws r"expected outgoing electron but got outgoing photon" PhaseSpacePoint(
                process, model, psl, in_particles_valid, (out_ph, out_el)
            )

            @test_throws r"expected 2 outgoing particles for the process but got 1" PhaseSpacePoint(
                process, model, psl, in_particles_valid, (out_el,)
            )

            @test_throws r"expected 2 incoming particles for the process but got 1" PhaseSpacePoint(
                process, model, psl, (out_el,), out_particles_valid
            )

            @test_throws r"expected 2 outgoing particles for the process but got 3" PhaseSpacePoint(
                process, model, psl, in_particles_valid, (out_el, out_el, out_ph)
            )

            @test_throws r"expected 2 incoming particles for the process but got 3" PhaseSpacePoint(
                process, model, psl, (in_el, in_el, in_ph), out_particles_valid
            )
        end

        @test_throws BoundsError momentum(psp, Incoming(), -1)
        @test_throws BoundsError momentum(psp, Outgoing(), -1)
        @test_throws BoundsError momentum(psp, Incoming(), 4)
        @test_throws BoundsError momentum(psp, Outgoing(), 4)

        @test_throws BoundsError psp[Incoming(), -1]
        @test_throws BoundsError psp[Outgoing(), -1]
        @test_throws BoundsError psp[Incoming(), 4]
        @test_throws BoundsError psp[Outgoing(), 4]

        @test_throws InvalidInputError PhaseSpacePoint(
            process, model, psl, in_particles_invalid, out_particles_valid
        )

        @test_throws InvalidInputError PhaseSpacePoint(
            process, model, psl, in_particles_valid, out_particles_invalid
        )

        @test_throws InvalidInputError PhaseSpacePoint(
            process, model, psl, (in_ph, in_el), out_particles_valid
        )

        @test_throws InvalidInputError PhaseSpacePoint(
            process, model, psl, in_particles_valid, (out_ph, out_el)
        )
    end

    @testset "Generation from momenta" begin
        test_psp = PhaseSpacePoint(
            process, model, psl, (in_el_mom, in_ph_mom), (out_el_mom, out_ph_mom)
        )

        @test test_psp.proc == process
        @test test_psp.model == model
        @test test_psp.psl == psl

        @test test_psp[Incoming(), 1] == in_el
        @test test_psp[Incoming(), 2] == in_ph
        @test test_psp[Outgoing(), 1] == out_el
        @test test_psp[Outgoing(), 2] == out_ph
    end

    @testset "Error handling from momenta" for (i, o) in
                                               Iterators.product([1, 3, 4, 5], [1, 3, 4, 5])
        @test_throws InvalidInputError PhaseSpacePoint(
            process,
            model,
            psl,
            TestImplementation._rand_momenta(RNG, i),
            TestImplementation._rand_momenta(RNG, o),
        )
    end

    @testset "Directional PhaseSpacePoint" begin
        @test psp isa PhaseSpacePoint
        @test psp isa InPhaseSpacePoint
        @test psp isa OutPhaseSpacePoint

        in_psp = InPhaseSpacePoint(process, model, psl, in_particles_valid)
        out_psp = OutPhaseSpacePoint(process, model, psl, out_particles_valid)
        in_psp_from_moms = InPhaseSpacePoint(process, model, psl, (in_el_mom, in_ph_mom))
        out_psp_from_moms = OutPhaseSpacePoint(
            process, model, psl, (out_el_mom, out_ph_mom)
        )

        @test in_psp isa InPhaseSpacePoint
        @test !(in_psp isa OutPhaseSpacePoint)
        @test in_psp_from_moms isa InPhaseSpacePoint
        @test !(in_psp_from_moms isa OutPhaseSpacePoint)

        @test out_psp isa OutPhaseSpacePoint
        @test !(out_psp isa InPhaseSpacePoint)
        @test out_psp_from_moms isa OutPhaseSpacePoint
        @test !(out_psp_from_moms isa InPhaseSpacePoint)

        @test_throws InvalidInputError InPhaseSpacePoint(
            process, model, psl, in_particles_invalid
        )
        @test_throws InvalidInputError OutPhaseSpacePoint(
            process, model, psl, out_particles_invalid
        )

        @testset "Error handling from momenta" for i in [1, 3, 4, 5]
            @test_throws InvalidInputError InPhaseSpacePoint(
                process, model, psl, TestImplementation._rand_momenta(RNG, i)
            )
            @test_throws InvalidInputError OutPhaseSpacePoint(
                process, model, psl, TestImplementation._rand_momenta(RNG, i)
            )
        end
    end
end
