# This can be removed, if https://github.com/QEDjl-project/QEDbase.jl/pull/129 is merged
# and released.

#######
# General coordinate transformations
#######

"""
    AbstractCoordinateTransformation

Abstract base type for coordinate transformations supposed to be acting on four-momenta.
Every subtype of `trafo::AbstractCoordinateTransformation` should implement the following interface functions:

* `QEDcore._transform(trafo,p)`: transforms `p`
* `Base.inv(trafo)`: returns the inverted transform

## Example

Implementing the interface by defining the interface functions:

```jldoctest trafo_interface
julia> using QEDcore

julia> struct TestTrafo{T} <: QEDcore.AbstractCoordinateTransformation
           a::T
       end

julia> QEDcore._transform(trafo::TestTrafo,p) = trafo.a*p

julia> Base.inv(trafo::TestTrafo) = TestTrafo(inv(trafo.a))

```

The `TestTrafo` can then be used to transform four-momenta:

```jldoctest trafo_interface
julia> trafo = TestTrafo(2.0)
TestTrafo{Float64}(2.0)

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> trafo(p) # multiply every component with 2.0
4-element SFourMomentum with indices SOneTo(4):
 8.0
 6.0
 4.0
 2.0

julia> inv(trafo)(p) # divide every component by 2.0
4-element SFourMomentum with indices SOneTo(4):
 2.0
 1.5
 1.0
 0.5
```
"""
abstract type AbstractCoordinateTransformation end
Base.broadcastable(trafo::AbstractCoordinateTransformation) = Ref(trafo)

"""
    _transform(trafo::AbstractCoordinateTransformation,p::AbstractFourMomentum)

Interface function for the application of the transformation to the four-momentum `p`. Must return a four-momentum
of the same type as `p`.
"""
function _transform end

# make the transform callable
@inline function (trafo::AbstractCoordinateTransformation)(p::AbstractFourMomentum)
    return _transform(trafo, p)
end

@inline function (trafo::AbstractCoordinateTransformation)(
    psf::PSF
) where {PSF<:AbstractParticleStateful}
    p_prime = _transform(trafo, momentum(psf))
    return PSF(p_prime)
end

@inline function (trafo::AbstractCoordinateTransformation)(
    psp::PSP
) where {PSP<:AbstractPhaseSpacePoint}
    in_moms = momenta(psp, Incoming())
    out_moms = momenta(psp, Outgoing())
    in_moms_prime = _transform.(trafo, in_moms)
    out_moms_prime = _transform.(trafo, out_moms)

    proc = process(psp)
    mod = model(psp)
    ps_def = phase_space_definition(psp)
    return PhaseSpacePoint(proc, mod, ps_def, in_moms_prime, out_moms_prime)
end

#########
# Abstract Lorentz Boosts
#########

"""

    AbstractLorentzTransformation <: AbstractCoordinateTransformation

An abstract base type representing Lorentz transformations, which are coordinate
transformations between inertial and reference frames in special relativity.

`AbstractLorentzTransformation` extends `AbstractCoordinateTransformation` and provides
the foundational framework for all types of Lorentz transformations, including boosts.
These transformations preserve the Minkowski product of two four-vectors and are fundamental to
the description of relativistic physics, ensuring the laws of physics are the same in all
inertial frames.

### Usage

Subtypes of `AbstractLorentzTransformation` implement specific kinds of Lorentz transformations.
For example:
- [`Boost{T}`](@ref): A concrete implementation of Lorentz boosts with boost parameter `T` (see also [`AbstractBoostParameter`](@ref)).

These subtypes perform transformations on four-vectors (such as [`SFourMomentum`](@ref)) between different inertial reference frames.
"""
abstract type AbstractLorentzTransformation <: AbstractCoordinateTransformation end

"""

    AbstractLorentzBoost <: AbstractLorentzTransformation

An abstract base type representing Lorentz boosts, a specific type of Lorentz transformation
associated with relative motion between inertial frames along one or more spatial directions.

`AbstractLorentzBoost` extends `AbstractLorentzTransformation` and serves as the foundation
for all types of boost transformations in special relativity. Lorentz boosts describe how
four-vectors (such as [`SFourMomentum`](@ref)) change when transitioning between two reference frames moving at constant velocities (in units of the speed of light) relative to each other.

For example:
- [`Boost{T}`](@ref): A concrete implementation of Lorentz boosts with boost parameter `T` (see also [`AbstractBoostParameter`](@ref)).

"""
abstract type AbstractLorentzBoost <: AbstractLorentzTransformation end

"""

    AbstractBoostParameter

An abstract base type representing boost parameters used in Lorentz transformations, which
describe the relative motion between two inertial frames in special relativity.

`AbstractBoostParameter` serves as the foundation for defining specific boost parameters
that control Lorentz boosts in different spatial directions. Boost parameters typically
represent the velocity of one reference frame relative to another, expressed as a fraction
of the speed of light (`\\beta`), and are essential for performing Lorentz transformations
on four-vectors (such as [`SFourMomentum`](@ref)).

## Overview

In the context of special relativity, a Lorentz boost is a transformation that changes the
time and spatial components of a four-vector based on the relative motion between two
inertial reference frames. For example, the boost parameter ``\\beta`` is dimensionless and represents
this velocity as a fraction of the speed of light. Depending on the frame's relative velocity,
different forms of boost parameters exist, such as those associated with a single axis or
a vector describing boosts in multiple spatial dimensions.

The `AbstractBoostParameter` type is the parent type for all specific kinds of boost parameters, including:
- **Axis-specific Boost Parameters**: Such as [`BetaX`](@ref), which describes a boost along the x-axis.
- **Vector-like Boost Parameters**: Such as [`BetaVector`](@ref), which describes boosts with components in multiple spatial directions.

"""
abstract type AbstractBoostParameter end

#############
# Interface: Phase space layout
#############

export AbstractPhaseSpaceLayout
export AbstractInPhaseSpaceLayout
export AbstractOutPhaseSpaceLayout
export build_momenta

abstract type AbstractPhaseSpaceLayout end
abstract type AbstractInPhaseSpaceLayout <: AbstractPhaseSpaceLayout end
abstract type AbstractOutPhaseSpaceLayout{InPSL<:AbstractInPhaseSpaceLayout} <:
              AbstractPhaseSpaceLayout end

"""

    phase_space_dimension(proc,model,::AbstractPhaseSpaceLayout)

TBW
"""
function phase_space_dimension end

"""

    in_phase_space_layout(::AbstactOutPhaseSpaceLayout)

TBW
"""
function in_phase_space_layout end

"""

    _build_momenta(proc,model,in_psl, in_coords)
    _build_momenta(proc,model,Ptot,out_psl,out_coords)

TBW
"""
function _build_momenta end

#############
# Implementations: Phase space layout
#############

### in ps layout

function _build_momenta(
    ::Val{Nc},
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    in_psl::AbstractInPhaseSpaceLayout,
    in_coords::NTuple{N},
) where {Nc,N}
    throw(
        InvalidInputError(
            "number of coordinates <$N> must be the same as the phase-space dimension <$Nc>"
        ),
    )
end

function _build_momenta(
    ::Val{Nc},
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    in_psl::AbstractInPhaseSpaceLayout,
    in_coords::NTuple{Nc,T},
) where {Nc,T}
    return _build_momenta(proc, model, in_psl, in_coords)
end

function build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    in_psl::AbstractInPhaseSpaceLayout,
    in_coords::Tuple,
)
    return _build_momenta(
        Val(phase_space_dimension(proc, model, in_psl)), proc, model, in_psl, in_coords
    )
end

"""
Scalar version of _build_momenta for in_psl
"""
function build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    in_psl::AbstractInPhaseSpaceLayout,
    in_coords::Real,
)
    return build_momenta(proc, model, in_psl, (in_coords,))
end

### out ps layout

function _build_momenta(
    ::Val{Nc},
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    Ptot::AbstractFourMomentum,
    out_psl::AbstractOutPhaseSpaceLayout,
    out_coords::NTuple{N},
) where {Nc,N}
    throw(
        InvalidInputError(
            "number of coordinates <$N> must be the same as the out-phase-space dimension <$Nc>",
        ),
    )
end

function _build_momenta(
    ::Val{Nc},
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    Ptot::AbstractFourMomentum,
    out_psl::AbstractOutPhaseSpaceLayout,
    out_coords::NTuple{Nc,T},
) where {Nc,T<:Real}
    return _build_momenta(proc, model, Ptot, out_psl, out_coords)
end

function build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    Ptot::AbstractFourMomentum,
    out_psl::AbstractOutPhaseSpaceLayout,
    out_coords::NTuple{Nc,T},
) where {Nc,T}
    return _build_momenta(
        Val(phase_space_dimension(proc, model, out_psl)),
        proc,
        model,
        Ptot,
        out_psl,
        out_coords,
    )
end
