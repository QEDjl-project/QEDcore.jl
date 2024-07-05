"""
    SphericalCoordinateSystem <: AbstractCoordinateSystem

TBW
"""
struct SphericalCoordinateSystem <: AbstractCoordinateSystem end

"""
    CenterOfMomentumFrame <: AbstractFrameOfReference

TBW
"""
struct CenterOfMomentumFrame <: AbstractFrameOfReference end

"""
    ElectronRestFrame <: AbstractFrameOfReference

TBW
"""
struct ElectronRestFrame <: AbstractFrameOfReference end

"""
    PhasespaceDefinition(coord_sys::AbstractCoordinateSystem, frame::AbstractFrameOfReference)

Convenient type to dispatch on coordiante systems and frames of reference. Combines a `AbstractCoordinateSystem` with a `AbstractFrameOfReference`.
"""
struct PhasespaceDefinition{CS<:AbstractCoordinateSystem,F<:AbstractFrameOfReference} <:
       AbstractPhasespaceDefinition
    coord_sys::CS
    frame::F
end

"""
    ParticleStateful <: AbstractParticle

Representation of a particle with a state. It has four fields:
- `dir::ParticleDirection`: The direction of the particle, `Incoming()` or `Outgoing()`.
- `species::AbstractParticleType`: The species of the particle, `Electron()`, `Positron()` etc.
- `mom::AbstractFourMomentum`: The momentum of the particle.

Overloads for `is_fermion`, `is_boson`, `is_particle`, `is_anti_particle`, `is_incoming`, `is_outgoing`, `mass`, and `charge` are provided, delegating the call to the correct field and thus implementing the `AbstractParticle` interface.

```jldoctest
julia> using QEDcore

julia> ParticleStateful(Incoming(), Electron(), SFourMomentum(1, 0, 0, 0))
ParticleStateful: incoming electron
    momentum: [1.0, 0.0, 0.0, 0.0]

julia> ParticleStateful(Outgoing(), Photon(), SFourMomentum(1, 0, 0, 0))
ParticleStateful: outgoing photon
    momentum: [1.0, 0.0, 0.0, 0.0]
```
"""
struct ParticleStateful{
    DIR<:ParticleDirection,SPECIES<:AbstractParticleType,ELEMENT<:AbstractFourMomentum
} <: AbstractParticleStateful{DIR,SPECIES,ELEMENT}
    dir::DIR
    species::SPECIES
    mom::ELEMENT

    function ParticleStateful(
        dir::DIR, species::SPECIES, mom::ELEMENT
    ) where {
        DIR<:ParticleDirection,SPECIES<:AbstractParticleType,ELEMENT<:AbstractFourMomentum
    }
        return new{DIR,SPECIES,ELEMENT}(dir, species, mom)
    end
end

"""
    PhaseSpacePoint

Representation of a point in the phase space of a process. Contains the process (`AbstractProcessDefinition`), the model (`AbstractModelDefinition`), the phase space definition (`AbstractPhasespaceDefinition`), and stateful incoming and outgoing particles ([`ParticleStateful`](@ref)).

The legality of the combination of the given process and the incoming and outgoing particles is checked on construction. If the numbers of particles mismatch, the types of particles mismatch (note that order is important), or incoming particles have an `Outgoing` direction, an error is thrown.

```jldoctest
julia> using QEDcore; using QEDprocesses

julia> PhaseSpacePoint(
            Compton(),
            PerturbativeQED(),
            PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
            (
                ParticleStateful(Incoming(), Electron(), SFourMomentum(1, 0, 0, 0)),
                ParticleStateful(Incoming(), Photon(), SFourMomentum(1, 0, 0, 0))
            ),
            (
                ParticleStateful(Outgoing(), Electron(), SFourMomentum(1, 0, 0, 0)),
                ParticleStateful(Outgoing(), Photon(), SFourMomentum(1, 0, 0, 0))
            )
        )
PhaseSpacePoint:
    process: one-photon Compton scattering
    model: perturbative QED
    phasespace definition: spherical coordinates in electron rest frame
    incoming particles:
     -> incoming electron: [1.0, 0.0, 0.0, 0.0]
     -> incoming photon: [1.0, 0.0, 0.0, 0.0]
    outgoing particles:
     -> outgoing electron: [1.0, 0.0, 0.0, 0.0]
     -> outgoing photon: [1.0, 0.0, 0.0, 0.0]
```

!!! note
    `PhaseSpacePoint`s can be constructed with only one of their in- or out-channel set. For this, see the special constructors [`InPhaseSpacePoint`](@ref) and [`OutPhaseSpacePoint`](@ref).
    The [`InPhaseSpacePoint`](@ref) and [`OutPhaseSpacePoint`](@ref) type definitions can be used to dispatch on such `PhaseSpacePoint`s. Note that a full `PhaseSpacePoint` containing both its in- and out-channel matches both, .i.e. `psp isa InPhaseSpacePoint` and `psp isa OutPhaseSpacePoint` both evaluate to true if psp contains both channels.
    A completely empty `PhaseSpacePoint` is not allowed.
"""
struct PhaseSpacePoint{
    PROC<:AbstractProcessDefinition,
    MODEL<:AbstractModelDefinition,
    PSDEF<:AbstractPhasespaceDefinition,
    IN_PARTICLES<:Tuple{Vararg{ParticleStateful}},
    OUT_PARTICLES<:Tuple{Vararg{ParticleStateful}},
    ELEMENT<:AbstractFourMomentum,
} <: AbstractPhaseSpacePoint{PROC,MODEL,PSDEF,IN_PARTICLES,OUT_PARTICLES}
    proc::PROC
    model::MODEL
    ps_def::PSDEF

    in_particles::IN_PARTICLES
    out_particles::OUT_PARTICLES

    """
        PhaseSpacePoint(
            proc::AbstractProcessDefinition, 
            model::AbstractModelDefinition, 
            ps_def::AbstractPhasespaceDefinition, 
            in_ps::Tuple{ParticleStateful},
            out_ps::Tuple{ParticleStateful},
        )

    Construct a [`PhaseSpacePoint`](@ref) from a process, model, phasespace definition and a tuple of [`ParticleStateful`](@ref)s.
    """
    function PhaseSpacePoint(
        proc::PROC, model::MODEL, ps_def::PSDEF, in_p::IN_PARTICLES, out_p::OUT_PARTICLES
    ) where {
        PROC<:AbstractProcessDefinition,
        MODEL<:AbstractModelDefinition,
        PSDEF<:AbstractPhasespaceDefinition,
        IN_PARTICLES<:Tuple{Vararg{ParticleStateful}},
        OUT_PARTICLES<:Tuple{Vararg{ParticleStateful}},
    }
        # this entire check is compiled away every time, so there's no need to disable it for performance ever
        ELEMENT = _check_psp(
            incoming_particles(proc), outgoing_particles(proc), in_p, out_p
        )

        return new{PROC,MODEL,PSDEF,IN_PARTICLES,OUT_PARTICLES,ELEMENT}(
            proc, model, ps_def, in_p, out_p
        )
    end
end

"""
    InPhaseSpacePoint

A partial type specialization on [`PhaseSpacePoint`](@ref) which can be used for dispatch in functions requiring only the in channel of the phase space to exist, for example implementations of `_incident_flux`. No restrictions are imposed on the out-channel, which may or may not exist.

See also: [`OutPhaseSpacePoint`](@ref)
"""
InPhaseSpacePoint{P,M,D,IN,OUT,E} = PhaseSpacePoint{
    P,M,D,IN,OUT,E
} where {IN<:Tuple{ParticleStateful,Vararg},OUT<:Tuple{Vararg}}

"""
    OutPhaseSpacePoint

A partial type specialization on [`PhaseSpacePoint`](@ref) which can be used for dispatch in functions requiring only the out channel of the phase space to exist. No restrictions are imposed on the in-channel, which may or may not exist.

See also: [`InPhaseSpacePoint`](@ref)
"""
OutPhaseSpacePoint{P,M,D,IN,OUT,E} = PhaseSpacePoint{
    P,M,D,IN,OUT,E
} where {IN<:Tuple{Vararg},OUT<:Tuple{ParticleStateful,Vararg}}
