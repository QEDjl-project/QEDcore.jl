#######
# Lorentz Boosts
#######
"""
TBW
"""
abstract type AbstractLorentzTransformation <: AbstractCoordinateTransformation end

"""
TBW
"""
abstract type AbstractLorentzBoost <: AbstractLorentzTransformation end

"""
TBW
"""
abstract type AbstractBoostParameter end

"""
TBW
"""
struct Boost{V<:AbstractBoostParameter} <: AbstractLorentzBoost
    param::V
end
boost_type(::Boost{V}) where {V} = V

# defaults
Boost(x::Real) = Boost(BetaX(x))
Boost(x::Real, y::Real, z::Real) = Boost(BetaVector(x, y, z))

# TODO:
# - add more convenient functions (type of the boost_param, ... )
# - interaction between several boosts? -> product trafo (for later)

function _transform(boost::Boost, p::AbstractFourMomentum)
    return _transform(boost.param, p)
end

function Base.inv(boost::Boost)
    return Boost(_inv(boost.param))
end
