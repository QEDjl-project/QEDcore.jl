# # Particles and Phase Space Points

# There are three layers of abstraction from particles to phase space points in the QEDjl project:
# - [`AbstractParticleType`](@extref QEDbase.AbstractParticleType): Base type for singleton particle type definitions. We also call these *species*.
# - [`AbstractParticleStateful`](@extref QEDbase.AbstractParticleStateful): Base type for particles with a direction and carrying a momentum.
# - [`AbstractPhaseSpacePoint`](@extref QEDbase.AbstractPhaseSpacePoint): Representation of a point in the phase space for a combination of an [`AbstractProcessDefinition`](@extref QEDbase.AbstractProcessDefinition), [`AbstractModelDefinition`](@extref QEDbase.AbstractModelDefinition), and [`AbstractPhasespaceDefinition`](@extref QEDbase.AbstractPhasespaceDefinition).

# This manual is intended to showcase the basic usage of these types and their implementations in QEDcore.

using QEDcore

# To use concrete process definitions and models, we also need to use [`QEDprocesses.jl`](https://github.com/QEDjl-project/QEDprocesses.jl)

using QEDprocesses

# ## Particle Types

# QEDcore currently defines the three basic particle types of QED [`Electron`](@ref), [`Positron`](@ref), and [`Photon`](@ref), and a type hierarchy for them:

@assert Photon <: MajoranaBoson
@assert Electron <: Fermion
@assert Positron <: AntiFermion

# There are also convenience functions in Julia convention:

@assert is_boson(Photon())
@assert is_particle(Electron())
@assert is_anti_particle(Positron())

# These functions are part of QEDbase.jl's [particle interface](@extref QEDbase particles).


