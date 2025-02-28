
###########
# Axis Beta
###########
"""

    AbstractAxisBeta{T} <: AbstractAxisBoostParameter{T}

An abstract base type for the beta (velocity) parameter of type `T`, representing a Lorentz boost along a specific spatial axis.

`AbstractAxisBeta{T}` extends `AbstractAxisBoostParameter{T}` and provides a general framework for defining beta parameters associated with individual Cartesian axes (x, y, z) in special relativity. The parameter `T` typically represents the numeric type (e.g., `Float64`, `Float32`) used for the beta value.

### Usage

Concrete subtypes of `AbstractAxisBeta{T}` define the beta parameters for Lorentz boosts along the x, y, and z axes:
- [`BetaX{T}`](@ref): Boost parameter for the x-axis.
- [`BetaY{T}`](@ref): Boost parameter for the y-axis.
- [`BetaZ{T}`](@ref): Boost parameter for the z-axis.

These beta parameters are essential for performing axis-specific Lorentz boosts, which transform physical quantities such as four-momentum between different inertial frames.

"""
abstract type AbstractAxisBeta{T} <: AbstractAxisBoostParameter{T} end

Base.:-(beta::B) where {B<:AbstractAxisBeta} = B(-beta.param)

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

Represents the beta parameter associated with a Lorentz boost along the x-axis, commonly denoted as ``\\beta_x``.

The transformation for a boost along the x-axis is:

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

julia> beta_x = BetaX(0.5)
BetaX{Float64}(0.5)

julia> boost = Boost(beta_x)
Boost{BetaX{Float64}}(BetaX{Float64}(0.5))

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 2.886751345948129
 1.1547005383792517
 2.0
 1.0

julia> @assert isapprox(p*p,p_prime*p_prime) # Invariant mass is preserved
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

function QEDbase._transform(boost_param::BetaX, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    px = getX(p)

    en_prime, px_prime = _generic_axis_boost(en, px, boost_param.param)
    return M(en_prime, px_prime, getY(p), getZ(p))
end

"""

    BetaY(beta::T) where {T<:Real}

Represents the beta parameter associated with a Lorentz boost along the y-axis, commonly denoted as ``\\beta_y``.

The transformation for a boost along the y-axis is:

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
4-element SFourMomentum{Float64} with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 3.4641016151377553
 3.0
 0.0
 1.0

julia> @assert isapprox(p*p,p_prime*p_prime) # Invariant mass is preserved
```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

"""
struct BetaY{T<:Real} <: AbstractAxisBeta{T}
    param::T
    function BetaY{T}(beta::T) where {T<:Real}
        -one(beta) <= beta < one(beta) ||
            throw(InvalidInputError("beta parameter <$beta> must be between zero and one"))
        return new{T}(beta)
    end
end

BetaY(beta::T) where {T} = BetaY{T}(beta)

function QEDbase._transform(boost_param::BetaY, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    py = getY(p)

    en_prime, py_prime = _generic_axis_boost(en, py, boost_param.param)
    return M(en_prime, getX(p), py_prime, getZ(p))
end

"""

    BetaZ(beta::T) where {T<:Real}

Represents the beta parameter associated with a Lorentz boost along the z-axis, commonly denoted as ``\\beta_z``.

The transformation for a boost along the z-axis is:

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
4-element SFourMomentum{Float64} with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum{Float64} with indices SOneTo(4):
  4.041451884327381
  3.0
  2.0
 -1.1547005383792517

julia> @assert isapprox(p*p,p_prime*p_prime) # Invariant mass is preserved

```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)

"""
struct BetaZ{T<:Real} <: AbstractAxisBeta{T}
    param::T
    function BetaZ{T}(beta::T) where {T<:Real}
        -one(beta) <= beta < one(beta) ||
            throw(InvalidInputError("beta parameter <$beta> must be between zero and one"))
        return new{T}(beta)
    end
end

BetaZ(beta::T) where {T} = BetaZ{T}(beta)
function QEDbase._transform(boost_param::BetaZ, p::M) where {M<:AbstractFourMomentum}
    en = getE(p)
    pz = getZ(p)

    en_prime, pz_prime = _generic_axis_boost(en, pz, boost_param.param)
    return M(en_prime, getX(p), getY(p), pz_prime)
end
