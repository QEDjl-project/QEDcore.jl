"""
    AbstractAxisBoostParameter{T}

An abstract base type representing a boost parameter of type `T`, associated with a specific axis in space (e.g., ``x``, ``y``, or ``z``).

This type serves as a foundation for concrete boost parameter types that define Lorentz boosts along individual spatial directions. The parameter `T` typically represents the data type for the boost value (e.g., `Float64`, `Float32`).

### Usage

Subtypes of `AbstractAxisBoostParameter{T}` are used to define specific boost transformations along a given axis (such as [`BetaX`](@ref) for the x-axis). These types are essential in performing Lorentz boosts, which transform four-momentum vectors between different inertial reference frames.

This abstract type is meant to be extended by concrete types to represent boosts along different Cartesian axes.

"""
abstract type AbstractAxisBoostParameter{T} <: QEDbase.AbstractBoostParameter end

function (::Type{BP})(boost_val::Real) where {T <: Real, BP <: AbstractAxisBoostParameter{T}}
    return BP(T(boost_val))
end

Base.eltype(::AbstractAxisBoostParameter{T}) where {T} = T
