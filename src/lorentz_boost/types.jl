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
    in_moms = momenta(psp,Incoming())
    out_moms = momenta(psp,Outgoing())
    in_moms_prime = _transform.(trafo, in_moms)
    out_moms_prime = _transform.(trafo, out_moms)

    proc = process(psp)
    mod = model(psp)
    ps_def = phase_space_definition(psp)
    return PhaseSpacePoint(proc,mod,ps_def,in_moms_prime,out_moms_prime)
end

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
