# # Particles and Phase Space Points

# There are three layers of abstraction from particles to phase space points in the QEDjl project:
# - [`QEDbase.AbstractParticleType`](@extref): Base type for singleton particle type definitions. We also call these *species*.
# - [`QEDbase.AbstractParticleStateful`](@extref): Base type for particles with a direction and carrying a momentum.
# - [`QEDbase.AbstractPhaseSpacePoint`](@extref): Representation of a point in the phase space for a combination of an [`QEDbase.AbstractProcessDefinition`](@extref), [`QEDbase.AbstractModelDefinition`](@extref), and [`QEDbase.AbstractPhasespaceDefinition`](@extref).

# This manual is intended to showcase the basic usage of these types and their implementations in QEDcore.

struct UnexpectedSuccess <: Exception end # hide
using QEDcore

# To use concrete process definitions and models, we also need to use [`QEDprocesses.jl`](https://github.com/QEDjl-project/QEDprocesses.jl)

using QEDprocesses

# ## Particle Types

# QEDcore currently defines the three basic particle types of QED, [`Electron`](@ref), [`Positron`](@ref), and [`Photon`](@ref), and a type hierarchy for them:

@assert Photon <: MajoranaBoson
@assert Electron <: Fermion
@assert Positron <: AntiFermion

# All of these are subtypes of [`QEDbase.AbstractParticleType`](@extref).
# There are also convenience functions in Julia convention:

@assert is_boson(Photon())
@assert is_particle(Electron())
@assert is_anti_particle(Positron())

@assert !is_boson(Electron())
@assert !is_anti_particle(Electron())
@assert !is_fermion(Photon())

# These functions are part of QEDbase.jl's [particle interface](@extref QEDbase Particle-Interface).

# ## ParticleStateful

# [`ParticleStateful`](@ref) is the implementation of QEDbase's [`QEDbase.AbstractParticleStateful`](@extref) interface. It represents a particle with a [`direction`](@extref QEDbase.ParticleDirection) (as used in the context of scattering processes, [`QEDbase.Incoming`](@extref), [`QEDbase.Outgoing`](@extref), or [`QEDbase.UnknownDirection`](@extref)), a particle species ([`Electron`](@ref), [`Positron`](@ref), [`Photon`](@ref), ...), and a 4-momentum vector.

ps = ParticleStateful(Incoming(), Electron(), rand(SFourMomentum))

# The relevant accessor functions for the interface are implemented:

particle_direction(ps)
#
particle_species(ps)
#
momentum(ps)
# 

# ## Phase Space Points

# A [`PhaseSpacePoint`](@ref) is the combination of incoming and outgoing [`ParticleStateful`](@ref)s. It also contains information about the [scattering process](@extref QEDbase.AbstractProcessDefinition), [model](@extref QEDbase.AbstractModelDefinition), and [phase space](@extref QEDbase.AbstractPhasespaceDefinition) that it is created for.

# ### Constructors

psp = PhaseSpacePoint(
    Compton(),                      # scattering process
    PerturbativeQED(),              # physics model
    PhasespaceDefinition(           # phase space definition
        SphericalCoordinateSystem(),# coordinate system
        ElectronRestFrame(),        # frame of reference
    ),
    (   # momenta of the incoming particles
        rand(SFourMomentum),
        rand(SFourMomentum),
    ),
    (   # momenta of the outgoing particles
        rand(SFourMomentum),
        rand(SFourMomentum),
    ),
)

# This version of the constructor automatically creates [`ParticleStateful`](@ref) obejcts from the momenta, matching the particles of the process. In the case of [`Compton`](@extref QEDprocesses.Compton), this is means an incoming electron and photon, and outgoing electron and photon.

# Automatic checks make sure that the number of 4-momenta given matches the necessary number of 4-momenta for the process (this adds 0 overhead at runtime because it is inferred from type information alone). 

try # hide
    PhaseSpacePoint(
        Compton(),
        PerturbativeQED(),
        PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
        (rand(SFourMomentum),), # incorrect number of incoming momenta, should be 2
        (rand(SFourMomentum), rand(SFourMomentum)),
    )
    throw(UnexpectedSuccess()) # hide
catch e # hide
    if e isa UnexpectedSuccess # hide
        rethrow(e) # hide
    end # hide
    @error e # hide
end # hide

# Alternatively, a [`PhaseSpacePoint`](@ref) can also be constructed from already existing [`ParticleStateful`](@ref) objects.

psp = PhaseSpacePoint(
    Compton(),                      # scattering process
    PerturbativeQED(),              # physics model
    PhasespaceDefinition(           # phase space definition
        SphericalCoordinateSystem(),# coordinate system
        ElectronRestFrame(),        # frame of reference
    ),
    (   # incoming particles
        ParticleStateful(Incoming(), Electron(), rand(SFourMomentum)),
        ParticleStateful(Incoming(), Photon(), rand(SFourMomentum)),
    ),
    (   # outgoing particles
        ParticleStateful(Outgoing(), Electron(), rand(SFourMomentum)),
        ParticleStateful(Outgoing(), Photon(), rand(SFourMomentum)),
    ),
)

# Similar to the constructor from momenta, this checks that the given [`ParticleStateful`](@ref)s fit to the given process and throws otherwise. Again, since this can be infered from type information alone, it adds no overhead.

try # hide
    PhaseSpacePoint(
        Compton(),
        PerturbativeQED(),
        PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
        (   # incoming particles
            ParticleStateful(Incoming(), Positron(), rand(SFourMomentum)), # incorrect particle type
            ParticleStateful(Incoming(), Photon(), rand(SFourMomentum)),
        ),
        (   # outgoing particles
            ParticleStateful(Outgoing(), Electron(), rand(SFourMomentum)),
            ParticleStateful(Outgoing(), Photon(), rand(SFourMomentum)),
        ),
    )
    throw(UnexpectedSuccess()) # hide
catch e # hide
    if e isa UnexpectedSuccess # hide
        rethrow(e) # hide
    end # hide
    @error e # hide
end # hide

# !!! note
#     While these constructors check that the given types make sense and work together, they do *not* check whether the given momenta make a physical phase space point or that the incoming or outgoing particles have on-shell 4-momenta.

# ### Accessors

# The phase space point provides some convenient accessors to the stateful particles within:

psp[Incoming(), 1]  # the first incoming particle
#
psp[Outgoing(), 2]  # the second outgoing particle
#
particles(psp, Incoming()) # all incoming particles as a tuple

# Momentum accessors:
momentum(psp, Incoming(), Electron(), 1) # the momentum of the first incoming electron

# When only one particle of the species exists in the particle set, the 1 can be ommitted for convenience.

@assert ans == momentum(psp, Incoming(), Electron())

# !!! note
#     This method throws when multiple (or zero) particles of the given direction and species exist in the phase space point.

# When the index of the required momentum is known at compile time, a `Val(N)` can be used instead of `N`. This performs bounds checks at compile time and removes loops from the runtime execution

using BenchmarkTools
judge(
    median(@benchmark momentum($psp, Incoming(), Photon(), Val(1))),
    median(@benchmark momentum($psp, Incoming(), Photon(), 1)),
)

# !!! note 
#     This is only faster when `N` is actually known at compile time, for example when it is a literal integer or a function's type parameter. For dynamic values of `N`, prefer the `Int` variant or in case of loops, directly loop over the tuple of [`momenta`](@extref QEDbase.momenta).

# Some more overloads for the momentum function exist, for a complete list please refer to its documentation: [`QEDbase.momentum`](@extref), [`QEDbase.momenta`](@extref).

# Finally, [`process`](@ref), [`model`](@ref), and [`phase_space_definition`](@ref) can be used to request the object in question:

process(psp)
#
model(psp)
#
phase_space_definition(psp)

# ## In/Out Phase Space Points

# As a special case, phase space points are allowed to only contain the incoming or outgoing particle momenta.
# These types can be helpful for overloading some functions that don't require the entire phase space point to exist.

function in_sum(in_psp::AbstractInPhaseSpacePoint)
    return sum(momenta(in_psp, Incoming()))
end

psp = InPhaseSpacePoint(
    Compton(),
    PerturbativeQED(),
    PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
    (rand(SFourMomentum), rand(SFourMomentum)),
)

in_sum(psp)

# Every full [`PhaseSpacePoint`](@ref) is both an [`InPhaseSpacePoint`](@ref) and an [`OutPhaseSpacePoint`](@ref), too. For example, the `in_sum` function defined above still works with a full [`PhaseSpacePoint`](@ref):

psp = PhaseSpacePoint(
    Compton(),
    PerturbativeQED(),
    PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
    (rand(SFourMomentum), rand(SFourMomentum)),
    (rand(SFourMomentum), rand(SFourMomentum)),
)

in_sum(psp)

# But an [`InPhaseSpacePoint`](@ref) is not an [`OutPhaseSpacePoint`](@ref) and vice versa. We cannot call `in_sum` on an [`OutPhaseSpacePoint`](@ref):

psp = OutPhaseSpacePoint(
    Compton(),
    PerturbativeQED(),
    PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
    (rand(SFourMomentum), rand(SFourMomentum)),
)

try # hide
    in_sum(psp)
    throw(UnexpectedSuccess()) # hide
catch e # hide
    if e isa UnexpectedSuccess # hide
        rethrow(e) # hide
    end # hide
    @error e # hide
end # hide
