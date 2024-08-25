"""
Return a random beta vector.
"""
function _rand_beta(rng::AbstractRNG, ::Type{T}=Float64) where {T<:Real}
    bx, by, bz = rand(rng, T, 3)
    r = sqrt(bx^2 + by^2 + bz^2)
    bxn, byn, bzn = bx / r, by / r, bz / r
    scale = 2 * rand(rng, T) - one(T)
    return BetaVector(bx * scale, by * scale, bz * scale)
end

@inline _rand(rng::AbstractRNG, ::Type{BetaVector}, ::Type{T}=Float64) where {T<:Real} =
    _rand_beta(rng, T)

function _rand(
    rng::AbstractRNG, ::Type{B}, ::Type{T}=Float64
) where {B<:QEDcore.AbstractAxisBeta,T<:Real}
    return B(2 * rand(rng, T) - one(T))
end
