
struct CenterOfMomentumEnergy <: AbstractUnivariateCoordinates end
const CMSEnergy = CenterOfMomentumEnergy
@inline coordinate_name(::CMSEnergy) = "cms_energy"

abstract type AbstractSingleParticleCoordinate{IDX} <: AbstractUnivariateCoordinates end
particle_index(::AbstractSingleParticleCoordinate{IDX}) where {IDX} = IDX

struct Energy{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline Energy(::Val{IDX}) where {IDX} = Energy{IDX}()
@inline Energy(idx::Int) = Energy(Val(idx))
@inline coordinate_name(c::Energy) = "energy_$(particle_index(c))"

struct Rapidity{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline Rapidity(::Val{IDX}) where {IDX} = Rapidity{IDX}()
@inline Rapidity(idx::Int) = Rapidity(Val(idx))
@inline coordinate_name(c::Rapidity) = "rapidity_$(particle_index(c))"

struct SpatialMagnitude{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline SpatialMagnitude(::Val{IDX}) where {IDX} = SpatialMagnitude{IDX}()
@inline SpatialMagnitude(idx::Int) = SpatialMagnitude(Val(idx))
@inline coordinate_name(c::SpatialMagnitude) = "spatial_magnitude_$(particle_index(c))"

struct CosTheta{IDX} <: AbstractSingleParticleCoordinate{IDX} end
@inline CosTheta(::Val{IDX}) where {IDX} = CosTheta{IDX}()
@inline CosTheta(idx::Int) = CosTheta(Val(idx))
@inline coordinate_name(c::CosTheta) = "cos_theta_$(particle_index(c))"
