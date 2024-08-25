#######
# General coordinate transformations
#######

"""
    AbstractCoordinateTransformation

Abstract base type for coordinate transformations supposed to be acting on four-momenta. 
Every subtype of `trafo::AbstractCoordianteTransformation` should implement the following interface functions:

* [`QEDcore._transform(trafo,p)`](@ref}: transfroms `p`
* `Base.inv(trafo)`: returns the inverted transform

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
    _transform(trafo,p)
end

@inline function (trafo::AbstractCoordinateTransformation)(psf::PSF) where {PSF<:ParticleStateful}
    p_prime = _transform(trafo,momentum(psf))
    return PSF(p_prime)
end

# FIXME: incoming and outgoing particles
@inline function (trafo::AbstractCoordinateTransformation)(psp::PSP) where {PSP<:PhasespacePoint}
    moms = momenta(psp)
    moms_prime = _transform.(trafo,moms)
    PSP(moms_prime)
end
# TODO: 
# - add convenient function `trafo(::ParticleStateful)`
# - add convenient function `trafo(::PhasespacePoint)`

#######
# Lorentz Boosts
#######
abstract type AbstractLorentzTransformation <: AbstractCoordinateTransformation end
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
boost_type(::Boost{V}) where V = V
Boost(x::Real) = Boost(BetaX(x))
Boost(x::Real,y::Real,z::Real) = Boost(BetaVector(x,y,z))

# TODO: 
# - add more convenient functions (type of the boost_param, ... )
# - interaction between several boosts? -> product trafo (for later)

function _transform(boost::Boost,p::AbstractFourMomentum)
    _transform(boost.param,p)
end

function Base.inv(boost::Boost)
    return Boost(_inv(boost.param))
end
