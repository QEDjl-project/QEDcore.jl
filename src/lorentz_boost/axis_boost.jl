abstract type AbstractAxisBoostParameter{T} <: AbstractBoostParameter end
convert(::Type{B}, param::S) where {T <: Real,B<:AbstractAxisBoostParameter{T},S <: Real} = B(T(param))
function Base.convert(::Type{B1}, d::B2) where {B1<:AbstractAxisBoostParameter,B2<:AbstractAxisBoostParameter}
    return B1(d.param)
end
Base.convert(::Type{B1}, d::B2) where {T<:Real,B1<:AbstractAxisBoostParameter{T},B2<:AbstractAxisBoostParameter{T}} = d

###########
# Axis Beta
###########
abstract type AbstractAxisBeta{T}<:AbstractAxisBoostParameter{T} end

-(beta::B) where {B<:AbstractAxisBeta} = B(-beta.param)

_inv(beta::B) where {B<:AbstractAxisBeta} = B(-beta.param)

@inline function _generic_axis_boost(en,comp,beta)
    b2 = beta^2
    gamma = inv(sqrt(one(b2) - b2))

    en_prime = gamma*(en - beta*comp)
    comp_prime = gamma*(comp - beta*en)

    return (en_prime, comp_prime)
end

"""
TBW
"""
struct BetaX{T} <: AbstractAxisBeta{T}
    param::T
    function BetaX{T}(beta::T) where {T}
        -one(beta)<=beta<one(beta) || throw(
            InvalidInputError(
                "beta parameter <$beta> must be between zero and one"
            )
        )
        return new{T}(beta)
    end
end

BetaX(beta::T) where T = BetaX{T}(beta)

function _transform(boost_param::BetaX,p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    px = getX(p)

    en_prime, px_prime = _generic_axis_boost(en,px,boost_param.param)
    return M(
        en_prime,
        px_prime,
        getY(p),
        getZ(p),
    )
end

"""
TBW
"""
struct BetaY{T} <: AbstractAxisBeta{T}
    param::T
    function BetaY{T}(beta::T) where {T}
        -one(beta)<=beta<one(beta) || throw(
            InvalidInputError(
                "beta parameter <$beta> must be between zero and one"
            )
        )
        return new{T}(beta)
    end
end

BetaY(beta::T) where T = BetaY{T}(beta)

function _transform(boost_param::BetaY,p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    py = getY(p)

    en_prime, py_prime = _generic_axis_boost(en,py,boost_param.param)
    return M(
        en_prime,
        getX(p),
        py_prime,
        getZ(p),
    )
end

"""
TBW
"""
struct BetaZ{T} <: AbstractAxisBeta{T}
    param::T
    function BetaZ{T}(beta::T) where {T}
        -one(beta)<=beta<one(beta) || throw(
            InvalidInputError(
                "beta parameter <$beta> must be between zero and one"
            )
        )
        return new{T}(beta)
    end
end

BetaZ(beta::T) where T = BetaZ{T}(beta)
function _transform(boost_param::BetaZ,p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    pz = getZ(p)

    en_prime, pz_prime = _generic_axis_boost(en,pz,boost_param.param)
    return M(
        en_prime,
        getX(p),
        getY(p),
        pz_prime,
    )
end


# TODO: 
# - add convenient constructors: Boost(:x,::Real)
# - add convenient constructors: Boost(:rest_frame,::AbstractFourMomentum)
# - add convenient constructors: Boost(::RestFrame,::AbstractFourMomentum)
