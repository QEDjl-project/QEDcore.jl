"""
    CenterOfMomentumEnergy <: AbstractUnivariateCoordinate

Represents the center-of-momentum (CoM) energy coordinate in the definion of a phase space
layout.
"""
struct CenterOfMomentumEnergy <: AbstractUnivariateCoordinate end
const CMSEnergy = CenterOfMomentumEnergy
@inline coordinate_name(::CMSEnergy) = "cms_energy"

"""
    AbstractSingleParticleCoordinate{IDX} <: AbstractUnivariateCoordinate

An abstract type representing a coordinate associated with a single particle.
The type parameter `IDX` indicates the index of the particle in an `AbstractProcessDefintion`,
either for the in- or out-channel, depending on the phase phase layout.
Specific types that inherit from this abstract type define various coordinates (e.g., energy, rapidity, etc.) for the particle.
"""
abstract type AbstractSingleParticleCoordinate{IDX} <: AbstractUnivariateCoordinate end

"""
    particle_index(coord::AbstractSingleParticleCoordinate{IDX})

Return the index of the particle the `coord` is related to, i.e. `IDX`.
"""
particle_index(::AbstractSingleParticleCoordinate{IDX}) where {IDX} = IDX

"""
    Energy{IDX} <: AbstractSingleParticleCoordinate{IDX}

Represents the energy coordinate for a single particle identified by `IDX`. This is mainly used
for multiple dispatch and the definiton of phase space layouts.
"""
struct Energy{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline Energy(::Val{IDX}) where {IDX} = Energy{IDX}()
@inline Energy(idx::Int) = Energy(Val(idx))
@inline coordinate_name(c::Energy) = "energy_$(particle_index(c))"

"""
    Rapidity{IDX} <: AbstractSingleParticleCoordinate{IDX}

Represents the rapidity coordinate for a single particle identified by `IDX`. This is mainly used
for multiple dispatch and the definiton of phase space layouts.
"""
struct Rapidity{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline Rapidity(::Val{IDX}) where {IDX} = Rapidity{IDX}()
@inline Rapidity(idx::Int) = Rapidity(Val(idx))
@inline coordinate_name(c::Rapidity) = "rapidity_$(particle_index(c))"

"""
    SpatialMagnitude{IDX} <: AbstractSingleParticleCoordinate{IDX}

Represents the spatial spatial magnitude for a single particle identified by `IDX`. This is mainly used
for multiple dispatch and the definiton of phase space layouts.
"""
struct SpatialMagnitude{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline SpatialMagnitude(::Val{IDX}) where {IDX} = SpatialMagnitude{IDX}()
@inline SpatialMagnitude(idx::Int) = SpatialMagnitude(Val(idx))
@inline coordinate_name(c::SpatialMagnitude) = "spatial_magnitude_$(particle_index(c))"

"""
    CosTheta{IDX} <: AbstractSingleParticleCoordinate{IDX}

Represents the cosine-theta coordinate for a single particle identified by `IDX`. This is mainly used
for multiple dispatch and the definiton of phase space layouts.
"""
struct CosTheta{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline CosTheta(::Val{IDX}) where {IDX} = CosTheta{IDX}()
@inline CosTheta(idx::Int) = CosTheta(Val(idx))
@inline coordinate_name(c::CosTheta) = "cos_theta_$(particle_index(c))"
