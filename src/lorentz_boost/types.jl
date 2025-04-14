"""
    Boost{V<:QEDbase.AbstractBoostParameter} <: QEDbase.AbstractLorentzBoost

A concrete type representing a Lorentz boost transformation, parameterized by a boost
parameter `V`. The boost parameter can be either axis-specific or vector-like, depending
on the subtype of `QEDbase.AbstractBoostParameter` used. The `Boost` type is used to perform
Lorentz boosts on four-vectors (such as four-momentum or four-position) between different
inertial frames in special relativity.

## Fields
- `param::V`: A boost parameter of type `V`, which is a subtype of `AbstractBoostParameter`.
    This parameter defines the velocity (as a fraction of the speed of light, ``\\beta``)
    and the direction of the boost (e.g., along a single axis or in multiple directions).

## Overview

A Lorentz boost is a transformation that adjusts the time and spatial components of a
four-vector based on the relative velocity between two reference frames. The `Boost`
struct provides a general and flexible implementation of such a boost, where the type of
the boost parameter determines the direction and magnitude of the boost.

Depending on the boost parameter `V`, the boost can be:
- **Axis-specific**: When `V` is an axis-specific boost parameter (e.g., `BetaX`), the
    boost will be along that axis.
- **Vector-like**: When `V` is a vector of boost parameters (e.g., `BetaVector`), the
    boost will have components in multiple spatial directions.

## Example
To create a Lorentz boost along the x-axis using the `BetaX` boost parameter:

```jldoctest example_boost_x
julia> using QEDcore

julia> beta_x = BetaX(0.5)
BetaX{Float64}(0.5)

julia> boost_x = Boost(beta_x)
Boost{BetaX{Float64}}(BetaX{Float64}(0.5))
```
To perform a Lorentz boost using the `boost_x` object, you can apply it to a four-vector,
such as four-momentum:

```jldoctest example_boost_x
julia> p = SFourMomentum(4, 3, 2, 1)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost_x(p)  # Perform the boost
4-element SFourMomentum{Float64} with indices SOneTo(4):
 2.886751345948129
 1.1547005383792517
 2.0
 1.0

julia> @assert isapprox(p*p, p_prime*p_prime)  # The invariant mass is preserved
```

## Notes

The `Boost` type provides a unified and flexible interface for applying Lorentz boosts,
with the boost parameter `V` determining the specific form of the transformation.
Lorentz boosts preserve the spacetime interval, meaning that applying the boost to a
four-vector will not change the invariant quantity.

## See Also

* `QEDbase.AbstractBoostParameter`: Base type for specific kinds of boost parameters.
* [`BetaX`](@ref): Boost parameter for the x-axis.
* [`BetaY`](@ref): Boost parameter for the y-axis.
* [`BetaZ`](@ref): Boost parameter for the z-axis.
* [`BetaVector`](@ref): Vector of boost parameters for boosts in multiple spatial directions.
"""
struct Boost{T <: QEDbase.AbstractBoostParameter} <: QEDbase.AbstractLorentzBoost
    param::T
end
boost_type(::Boost{T}) where {T} = T
Base.eltype(boost::Boost) = eltype(boost.param)

# defaults
Boost(x::Real) = Boost(BetaX(x))
Boost(x::Real, y::Real, z::Real) = Boost(BetaVector(x, y, z))

function QEDbase._transform(boost::Boost, p::AbstractFourMomentum)
    return QEDbase._transform(boost.param, p)
end

function Base.inv(boost::Boost)
    return Boost(_inv(boost.param))
end

"""
    _unsafe_rest_boost(p::AbstractFourMomentum)

Compute the boost transformation to the rest frame of the specified momentum `p`, such that `boost(p) == SFourMomentum(getMass(p), 0, 0, 0)`.

## Notes

This function is considered "unsafe" because it does not verify that `p` has a non-zero mass, which is required for the existence of a valid rest frame.
"""
function _unsafe_rest_boost(p::AbstractFourMomentum)
    E = @inbounds p[1]
    beta_val = SVector{3}(view(p, 2:4)) / E
    beta = @inbounds BetaVector(beta_val...)
    return Boost(beta)
end
