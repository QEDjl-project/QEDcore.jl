
# TODO: 
# - add interaction with axis boosts
# - add convenient constructors BetaVector(p) for the rest system?
abstract type AbstractBoostVector <: AbstractBoostParameter end

"""
TBW
"""
struct BetaVector{T<:Real} <: AbstractBoostVector
    x::T
    y::T
    z::T

    function BetaVector(x::T,y::T,z::T) where {T}
        b2 = x^2 + y^2 + z^2
         b2<=1 || throw(
            InvalidInputError(
                "wrong length of the beta vector ($x, $y, $z). Its length needs to be less or equal to one, but x^2 + y^2 + z^2 = $b2 is given."
            )
        )
        new{T}(x,y,z)
    end
end
BetaVector(x,y,z) = BetaVector(promote(x,y,z)...)

import Base:-

-(b::BetaVector) = BetaVector(-b.x,-b.y,-b.z)

function Base.isapprox(b1::BetaVector,b2::BetaVector;
                  atol::Real=0, rtol::Real=Base.rtoldefault(b1.x,b1.y,atol),
                  nans::Bool=false, norm::Function=abs)
    isapprox(b1.x,b2.x;atol=atol,rtol=rtol,nans=nans,norm=norm) &&
    isapprox(b1.y,b2.y;atol=atol,rtol=rtol,nans=nans,norm=norm) &&
    isapprox(b1.z,b2.z;atol=atol,rtol=rtol,nans=nans,norm=norm) 
end

@inline function _mul(p::AbstractFourMomentum,beta::BetaVector)
    p[2]*beta.x + p[3]*beta.y + p[4]*beta.z
end
_mul(beta::BetaVector,p::AbstractFourMomentum) = _mul(p,beta)

function _square(beta_vec::BetaVector)
    bx = beta_vec.x
    by = beta_vec.y
    bz = beta_vec.z
    bx^2 + by^2 + bz^2
end


@inline function _transform(beta_vec::BetaVector,p::M) where {M<:AbstractFourMomentum}
    b2 = _square(beta_vec)
    if b2 == one(b2)
        return p
    end
    gamma = inv(sqrt(one(b2) - b2))

    en = getE(p)
    px = getX(p)
    py = getY(p)
    pz = getZ(p)
    bp = _mul(p,beta_vec)
    gamma2 = (gamma - one(b2))/b2 
    fac = gamma2*bp - gamma*en 
    px_prime = px + fac*beta_vec.x
    py_prime = py + fac*beta_vec.y 
    pz_prime = pz + fac*beta_vec.z 
    en_prime = gamma*(en - bp)

    return M(
        en_prime,
        px_prime,
        py_prime,
        pz_prime,
    )
end

_inv(beta_vec::BetaVector) = BetaVector(-beta_vec.x,-beta_vec.y,-beta_vec.z)
