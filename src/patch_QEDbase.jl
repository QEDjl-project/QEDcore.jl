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

"""
    AbstractPhaseSpaceLayout

The `AbstractPhaseSpaceLayout` is an abstract type that represents the general concept of
a phase space layout in a scattering process.

## Interface Functions to Implement:
- `phase_space_dimension(proc, model, layout::AbstractPhaseSpaceLayout)`: Defines the number
    of independent phase space coordinates needed to build the momenta.
"""
abstract type AbstractPhaseSpaceLayout end

"""
    AbstractInPhaseSpaceLayout <: AbstractPhaseSpaceLayout

The `AbstractInPhaseSpaceLayout` represents the phase space layout for the incoming particles
in a scattering process. It defines the way in which the momenta of incoming particles are
constructed from phase space coordinates.

## Interface Functions to Implement:
- `phase_space_dimension(proc, model, layout::AbstractInPhaseSpaceLayout)`: Defines the number
    of independent phase space coordinates for the incoming particles.
- `build_momenta(proc, model, in_psl::AbstractInPhaseSpaceLayout, in_coords::Tuple)`: Constructs
    the momenta for the incoming particles using the phase space coordinates.
"""
abstract type AbstractInPhaseSpaceLayout <: AbstractPhaseSpaceLayout end

"""
    AbstractOutPhaseSpaceLayout{InPSL<:AbstractInPhaseSpaceLayout} <: AbstractPhaseSpaceLayout

The `AbstractOutPhaseSpaceLayout` represents the phase space layout for the outgoing particles
in a scattering process. It typically depends on the phase space layout of the incoming particles,
and it specifies how the momenta of the outgoing particles are constructed from the respective coordinates.

The generic parameter `InPSL` links the outgoing phase space layout to the incoming layout,
allowing consistency between the two configurations in the process.

## Interface Functions to Implement:
- `phase_space_dimension(proc, model, layout::AbstractOutPhaseSpaceLayout)`: Defines the
    number of independent phase space coordinates for the outgoing particles.
- `in_phase_space_layout(out_psl::AbstractOutPhaseSpaceLayout)`: Provides the associated
    incoming phase space layout to ensure consistency between incoming and outgoing configurations.
- `build_momenta(proc, model, Ptot::AbstractFourMomentum, out_psl::AbstractOutPhaseSpaceLayout, out_coords::Tuple)`:
    Constructs the momenta for the outgoing particles, ensuring they comply with energy and momentum conservation based on the total incoming four-momentum.
"""
abstract type AbstractOutPhaseSpaceLayout{InPSL<:AbstractInPhaseSpaceLayout} <:
              AbstractPhaseSpaceLayout end
"""
    phase_space_dimension(proc, model, layout::AbstractPhaseSpaceLayout) -> Int

This function needs to be implemented for the phase-space layout interface.
Return the dimensionality of the phase space, i.e. the numebr of coordinates, for a given
process and model within the specified `layout`.

The phase space dimension is a crucial quantity that determines how many independent coordinates
are required to describe the system of particles in the scattering process. It depends on
the number of particles involved and the specific interaction model in use.

## Arguments
- `proc`: The scattering process definition, a subtype of `AbstractProcessDefinition`.
- `model`: The physics model, a subtype of `AbstractModelDefinition`.
- `layout`: A specific phase space layout, either `AbstractInPhaseSpaceLayout` or
    `AbstractOutPhaseSpaceLayout`.

## Returns
- The integer representing the number of independent phase space coordinates.
"""
function phase_space_dimension end

"""
    in_phase_space_layout(out_psl::AbstractOutPhaseSpaceLayout) -> AbstractInPhaseSpaceLayout

This function needs to be implemented for the out-phase-space layout interface.
Given an outgoing phase space layout (`out_psl`), this function returns the associated incoming
phase space layout.

This is useful for ensuring consistency between the incoming and outgoing particle momenta
when calculating or sampling phase space points in scattering processes.


## Arguments
- `out_psl`: The outgoing phase space layout, a subtype of `AbstractOutPhaseSpaceLayout`.

## Returns
- The associated incoming phase space layout, a subtype of `AbstractInPhaseSpaceLayout`.
"""
function in_phase_space_layout end

"""

    _build_momenta(proc,model,in_psl, in_coords)
    _build_momenta(proc,model,Ptot,out_psl,out_coords)

TBW
"""

"""
    _build_momenta(proc, model, in_psl::AbstractInPhaseSpaceLayout, in_coords::Tuple)
    _build_momenta(proc, model, Ptot::AbstractFourMomentum, out_psl::AbstractOutPhaseSpaceLayout, out_coords::Tuple)

These functions need to be implemented as part of the phase space layout interface for both
incoming and outgoing particle momenta construction. They serve as internal, low-level
interfaces for constructing the four-momenta of particles during a scattering process, and
are typically wrapped by the user-facing [`build_momenta`](@ref) function.

### Incoming Phase Space Layout
The first function, `_build_momenta(proc, model, in_psl, in_coords)`, constructs the
four-momenta for the incoming particles based on the specified phase space coordinates
(`in_coords`).

- **Arguments**:
    - `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
    - `model`: The physics model, subtype of `AbstractModelDefinition`.
    - `in_psl`: The incoming phase space layout, subtype of `AbstractInPhaseSpaceLayout`,
        that defines how to map the coordinates to momenta.
    - `in_coords`: A tuple of phase space coordinates that parametrize the momenta of the
        incoming particles.

- **Returns**:
    - A collection of four-momenta representing the incoming particles. For performance reasons,
        it is recommended to return a `Tuple` of four-momenta.

### Outgoing Phase Space Layout
The second function, `_build_momenta(proc, model, Ptot, out_psl, out_coords)`, constructs the
four-momenta for the outgoing particles. It uses the total four-momentum (`Ptot`) from the
incoming state and applies the phase space coordinates (`out_coords`) to compute the outgoing
momenta, ensuring they adhere to energy and momentum conservation laws.

- **Arguments**:
    - `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
    - `model`: The physics model, subtype of `AbstractModelDefinition`.
    - `Ptot`: The total incoming four-momentum, which is used to compute the momenta of the
        outgoing particles.
    - `out_psl`: The outgoing phase space layout, subtype of `AbstractOutPhaseSpaceLayout`,
        that maps the coordinates to momenta.
    - `out_coords`: A tuple of phase space coordinates that parametrize the outgoing particle
        momenta.

- **Returns**:
    - A collection of four-momenta representing the outgoing particles. For performance reasons,
        it is recommended to return a `Tuple` of four-momenta.

### Notes
Both versions of `_build_momenta` handle the construction of particle momenta during different
phases of the scattering process:
- The **incoming version** constructs momenta based on phase space coordinates alone.
- The **outgoing version** constructs momenta based on both phase space coordinates and
    the total four-momentum, ensuring conservation laws are respected.
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

"""
    build_momenta(proc, model, in_psl::AbstractInPhaseSpaceLayout, in_coords::Tuple)

Construct the momenta of the incoming particles using the provided phase space coordinates.

This is the user-facing function that calls `_build_momenta` internally and validates the
number of coordinates against the phase space dimensionality.

# Arguments
- `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
- `model`: The physics model, subtype of `AbstractModelDefinition`.
- `in_psl`: The incoming phase space layout, subtype of `AbstractInPhaseSpaceLayout`.
- `in_coords`: A tuple of phase space coordinates that parametrize the incoming particle momenta.

# Returns
- A collection of four-momenta representing the incoming particles. Because of performance
    reasons, it is recommened to return a `Tuple` of four-momenta.
"""
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
    build_momenta(proc::AbstractProcessDefinition, model::AbstractModelDefinition, in_psl::AbstractInPhaseSpaceLayout, in_coords::Real)

A scalar version of `build_momenta` for incoming phase space layouts (`in_psl`), where the phase space coordinates are provided as a single scalar instead of a tuple.

## Arguments:
- `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
- `model`: The physics model, subtype of `AbstractModelDefinition`.
- `in_psl`: The incoming phase space layout, subtype of `AbstractInPhaseSpaceLayout`.
- `in_coords::Real`: A single scalar representing the phase space coordinate for the
    incoming particles.

## Returns:
- A collection of four-momenta representing the incoming particles. Because of performance
    reasons, it is recommened to return a `Tuple` of four-momenta.

## Notes:
This function is a convenience wrapper around `build_momenta`, automatically converting the
scalar `in_coords` into a 1-tuple. It is useful when the incoming phase space only requires
a single coordinate to define the particle momenta.

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

"""
    build_momenta(proc, model, Ptot::AbstractFourMomentum, out_psl::AbstractOutPhaseSpaceLayout, out_coords::Tuple)

Construct the momenta of the outgoing particles using the provided phase space coordinates (`out_coords`)
and total incoming momentum (`Ptot`).

This function ensures that the outgoing momenta satisfy energy and momentum conservation,
consistent with the physics model in use.

# Arguments
- `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
- `model`: The physics model, subtype of `AbstractModelDefinition`.
- `Ptot`: The total incoming four-momentum, used to compute the outgoing momenta.
- `out_psl`: The outgoing phase space layout, subtype of `AbstractOutPhaseSpaceLayout.
- `out_coords`: A tuple of phase space coordinates that parametrize the outgoing particle momenta.

# Returns
- A collection of four-momenta representing the incoming particles. Because of performance
    reasons, it is recommened to return a `Tuple` of four-momenta.
"""
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
