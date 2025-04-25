"""

    BetaVector(x::Real,y::Real,z::Real)

Represents the spatial vector of velocity parameters (denoted as the "beta" vector) associated with motion in the three Cartesian directions, i.e.,
``\\vec\\beta = (\\beta_x, \\beta_y, \\beta_z)``. These components correspond to the velocity of an object (in units of the speed of light) in each of the
``x``, ``y``, and ``z`` directions.

The Lorentz boost along the direction of the beta vector ``\\vec\\beta`` transforms the four-momentum as follows:

```math
\\begin{pmatrix}
p_0\\\\
p_1\\\\
p_2\\\\
p_3
\\end{pmatrix} \\mapsto
\\begin{pmatrix}
    \\gamma  (p_0 - \\vec\\beta \\vec p)\\\\
p_1 + (\\frac{\\gamma - 1}{\\beta^2} \\vec\\beta\\vec p - \\gamma  p_0)
 \\beta_x\\\\
p_2 + (\\frac{\\gamma - 1}{\\beta^2} \\vec\\beta\\vec p - \\gamma  p_0)
 \\beta_y\\\\
    p_3 + (\\frac{\\gamma - 1}{\\beta^2} \\vec\\beta\\vec p - \\gamma  p_0)
 \\beta_z\\\\
\\end{pmatrix}
```
where the kinematic factor is given as ``\\gamma = 1/\\sqrt{1-\\beta_x^2}``.

## Example

```jldoctest
julia> using QEDcore

julia> beta_vec = BetaVector(0.2,0.3,0.1)
BetaVector{Float64}(0.2, 0.3, 0.1)

julia> boost = Boost(beta_vec)
Boost{BetaVector{Float64}}(BetaVector{Float64}(0.2, 0.3, 0.1))

julia> p = SFourMomentum(4.0,3.0,2.0,1.0)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> p_prime = boost(p)
4-element SFourMomentum{Float64} with indices SOneTo(4):
 2.911484876492837
 2.282803602436349
 0.9242054036545237
 0.6414018012181746

julia> @assert isapprox(p*p,p_prime*p_prime) # Invariant mass is preserved
```

## External link

* [Lorentz Boost on Wikipedia](https://en.wikipedia.org/wiki/Lorentz_transformation)
* [Kinematics in PDG review](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf)
* [`ROOT::Math:Boost` from ROOT](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1Boost.html)

"""
struct BetaVector{T <: Real} <: AbstractBoostVector
    x::T
    y::T
    z::T

    function BetaVector(x::T, y::T, z::T) where {T}
        b2 = x^2 + y^2 + z^2
        b2 <= 1 || throw(
            InvalidInputError(
                "wrong length of the beta vector, its length needs to be less or equal to one",
            ),
        )
        return new{T}(x, y, z)
    end
end
BetaVector(x, y, z) = BetaVector(promote(x, y, z)...)

Base.:-(b::BetaVector) = BetaVector(-b.x, -b.y, -b.z)

@inline function _spatial_mul(p::AbstractFourMomentum, beta::BetaVector)
    return p[2] * beta.x + p[3] * beta.y + p[4] * beta.z
end

# assumption: beta vector components commute with four-momentum components
_spatial_mul(beta::BetaVector, p::AbstractFourMomentum) = _spatial_mul(p, beta)

function _three_vector_square(beta_vec::BetaVector)
    bx = beta_vec.x
    by = beta_vec.y
    bz = beta_vec.z
    return bx^2 + by^2 + bz^2
end

@inline function QEDbase._transform(
        beta_vec::BetaVector, p::M
    ) where {M <: AbstractFourMomentum}
    b2 = _three_vector_square(beta_vec)
    if b2 == one(b2)
        return p
    end
    gamma = inv(sqrt(one(b2) - b2))

    en = getE(p)
    px = getX(p)
    py = getY(p)
    pz = getZ(p)
    bp = _spatial_mul(p, beta_vec)
    gamma2 = (gamma - one(b2)) / b2
    fac = gamma2 * bp - gamma * en
    px_prime = px + fac * beta_vec.x
    py_prime = py + fac * beta_vec.y
    pz_prime = pz + fac * beta_vec.z
    en_prime = gamma * (en - bp)

    return M(en_prime, px_prime, py_prime, pz_prime)
end

# inverse is just a boost with -beta
_inv(beta_vec::BetaVector) = BetaVector(-beta_vec.x, -beta_vec.y, -beta_vec.z)
