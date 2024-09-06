# TODO: 
# - test conversions
# - add convenient constructors: Boost(:x,::Real)
# - add convenient constructors: Boost(:rest_frame,::AbstractFourMomentum)
# - add convenient constructors: Boost(::RestFrame,::AbstractFourMomentum)

"""

    AbstractAxisBoostParameter{T}

Abstract base type for boost parameter of type `T` associated to a certain axis in space.
"""
abstract type AbstractAxisBoostParameter{T} <: AbstractBoostParameter end
function convert(
    ::Type{B}, param::S
) where {T<:Real,B<:AbstractAxisBoostParameter{T},S<:Real}
    return B(T(param))
end
function Base.convert(
    ::Type{B1}, d::B2
) where {B1<:AbstractAxisBoostParameter,B2<:AbstractAxisBoostParameter}
    return B1(d.param)
end
function Base.convert(
    ::Type{B1}, d::B2
) where {T<:Real,B1<:AbstractAxisBoostParameter{T},B2<:AbstractAxisBoostParameter{T}}
    return d
end

###########
# Axis Beta
###########
"""

    AbstractAxisBeta{T} <: AbstractAxisBoostParameter{T}

Abstact base type for boost beta parameter of type `T` associated to an axis in space.
"""
abstract type AbstractAxisBeta{T} <: AbstractAxisBoostParameter{T} end

-(beta::B) where {B<:AbstractAxisBeta} = B(-beta.param)

_inv(beta::B) where {B<:AbstractAxisBeta} = B(-beta.param)

@inline function _generic_axis_boost(en, comp, beta)
    b2 = beta^2
    gamma = inv(sqrt(one(b2) - b2))

    en_prime = gamma * (en - beta * comp)
    comp_prime = gamma * (comp - beta * en)

    return (en_prime, comp_prime)
end

"""

    BetaX(beta::T) where {T<:Real}

Beta parameter associated to the x-axis, commonly referred to as ``\\beta_x``. 
The corresponding Lorentz boost reads

```math
\\begin{pmatrix}
p_0\\\\
p_1\\\\
p_2\\\\
p_3
\\end{pmatrix} \\mapsto
\\begin{pmatrix}
\\gamma (p_0 - \\beta_x p_1)\\\\
\\gamma (p_1 - \\beta_x p_0)\\\\
p_2\\\\
p_3
\\end{pmatrix}
```
where the kinematic factor is given as ``\\gamma = 1/\\sqrt{1-\\beta_x^2}``)

## Example

```jldoctest
julia> using QEDcore

julia> using Random

julia> RNG = MersenneTwister(1234)
MersenneTwister(1234)

julia> beta_x = BetaX(0.5)
BetaX{Float64}(0.5)

julia> boost = Boost(beta_x)
Boost{BetaX{Float64}}(BetaX{Float64}(0.5))

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum with indices SOneTo(4):
 2.886751345948129
 1.1547005383792517
 2.0
 1.0

julia> @assert isapprox(p*p,p_prime*p_prime) 
```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

"""
struct BetaX{T<:Real} <: AbstractAxisBeta{T}
    param::T
    function BetaX{T}(beta::T) where {T<:Real}
        -one(beta) <= beta < one(beta) ||
            throw(InvalidInputError("beta parameter <$beta> must be between zero and one"))
        return new{T}(beta)
    end
end

BetaX(beta::T) where {T<:Real} = BetaX{T}(beta)

function _transform(boost_param::BetaX, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    px = getX(p)

    en_prime, px_prime = _generic_axis_boost(en, px, boost_param.param)
    return M(en_prime, px_prime, getY(p), getZ(p))
end

"""

    BetaY(beta::T) where {T<:Real}

Beta parameter associated to the y-axis, commonly referred to as ``\\beta_y``. 
The corresponding Lorentz boost reads

```math
\\begin{pmatrix}
p_0\\\\
p_1\\\\
p_2\\\\
p_3
\\end{pmatrix} \\mapsto
\\begin{pmatrix}
\\gamma (p_0 - \\beta_y p_2)\\\\
p_1\\\\
\\gamma (p_2 - \\beta_y p_0)\\\\
p_3
\\end{pmatrix}
```
where the kinematic factor is given as ``\\gamma = 1/\\sqrt{1-\\beta_y^2}``)

## Example

```jldoctest
julia> using QEDcore

julia> using Random

julia> RNG = MersenneTwister(1234)
MersenneTwister(1234)

julia> beta_y = BetaY(0.5)
BetaY{Float64}(0.5)

julia> boost = Boost(beta_y)
Boost{BetaY{Float64}}(BetaY{Float64}(0.5))

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum with indices SOneTo(4):
 3.4641016151377553
 3.0
 0.0
 1.0

julia> @assert isapprox(p*p,p_prime*p_prime) 
```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

"""
struct BetaY{T} <: AbstractAxisBeta{T}
    param::T
    function BetaY{T}(beta::T) where {T}
        -one(beta) <= beta < one(beta) ||
            throw(InvalidInputError("beta parameter <$beta> must be between zero and one"))
        return new{T}(beta)
    end
end

BetaY(beta::T) where {T} = BetaY{T}(beta)

function _transform(boost_param::BetaY, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    py = getY(p)

    en_prime, py_prime = _generic_axis_boost(en, py, boost_param.param)
    return M(en_prime, getX(p), py_prime, getZ(p))
end

"""

    BetaZ(beta::T) where {T<:Real}

Beta parameter associated to the z-axis, commonly referred to as ``\\beta_z``. 
The corresponding Lorentz boost reads

```math
\\begin{pmatrix}
p_0\\\\
p_1\\\\
p_2\\\\
p_3
\\end{pmatrix} \\mapsto
\\begin{pmatrix}
\\gamma (p_0 - \\beta_z p_3)\\\\
p_1\\\\
p_2\\\\
\\gamma (p_3 - \\beta_z p_0)\\\\
\\end{pmatrix}
```
where the kinematic factor is given as ``\\gamma = 1/\\sqrt{1-\\beta_z^2}``)

## Example

```jldoctest
julia> using QEDcore

julia> using Random

julia> RNG = MersenneTwister(1234)
MersenneTwister(1234)

julia> beta_z = BetaZ(0.5)
BetaZ{Float64}(0.5)

julia> boost = Boost(beta_z)
Boost{BetaZ{Float64}}(BetaZ{Float64}(0.5))

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum with indices SOneTo(4):
  4.041451884327381
  3.0
  2.0
 -1.1547005383792517

julia> @assert isapprox(p*p,p_prime*p_prime) 

```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

"""
struct BetaZ{T} <: AbstractAxisBeta{T}
    param::T
    function BetaZ{T}(beta::T) where {T}
        -one(beta) <= beta < one(beta) ||
            throw(InvalidInputError("beta parameter <$beta> must be between zero and one"))
        return new{T}(beta)
    end
end

BetaZ(beta::T) where {T} = BetaZ{T}(beta)
function _transform(boost_param::BetaZ, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    pz = getZ(p)

    en_prime, pz_prime = _generic_axis_boost(en, pz, boost_param.param)
    return M(en_prime, getX(p), getY(p), pz_prime)
end
