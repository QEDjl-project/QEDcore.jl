"""
Return a random beta vector.
"""
function _rand_beta(rng::AbstractRNG, ::Type{T} = Float64) where {T <: Real}
    beta_xyz = rand(rng, T, 3)
    beta_xyz .*= 2
    beta_xyz .-= 1
    beta_xyz ./= sqrt(3)
    return BetaVector(beta_xyz...)
end

@inline _rand(rng::AbstractRNG, ::Type{BetaVector}, ::Type{T} = Float64) where {T <: Real} =
    _rand_beta(rng, T)

function _rand(
        rng::AbstractRNG, ::Type{B}, ::Type{T} = Float64
    ) where {B <: QEDcore.AbstractAxisBeta, T <: Real}
    return B(2 * rand(rng, T) - one(T))
end
