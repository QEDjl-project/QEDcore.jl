
# Check if any failed type is in the input
_any_fail(x...) = true
_any_fail(::TestProcess, ::TestModel) = false
_any_fail(::TestProcess, ::TestModel, ::TestPhasespaceDef) = false

# unrolls all elements of a list of four-momenta into vector of coordinates
function _unroll_moms(ps_moms::AbstractVector{T}) where {T<:AbstractFourMomentum}
    return collect(Iterators.flatten(ps_moms))
end

function _unroll_moms(ps_moms::AbstractMatrix{T}) where {T<:AbstractFourMomentum}
    res = Matrix{eltype(T)}(undef, size(ps_moms, 1) * 4, size(ps_moms, 2))
    for i in 1:size(ps_moms, 2)
        res[:, i] .= _unroll_moms(view(ps_moms, :, i))
    end
    return res
end

flat_components(moms::AbstractVecOrMat) = _unroll_moms(moms)
flat_components(moms::Tuple) = Tuple(_unroll_moms([moms...]))

# collect components of four-momenta from a vector of coordinates
function __furl_moms(ps_coords::AbstractVector{T}) where {T<:Real}
    return SFourMomentum.(eachcol(reshape(ps_coords, 4, :)))
end

function _furl_moms(ps_coords::AbstractVector{T}) where {T<:Real}
    @assert length(ps_coords) % 4 == 0
    return __furl_moms(ps_coords)
end

function _furl_moms(ps_coords::AbstractMatrix{T}) where {T<:Real}
    @assert size(ps_coords, 1) % 4 == 0
    res = Matrix{SFourMomentum}(undef, Int(size(ps_coords, 1)//4), size(ps_coords, 2))
    for i in 1:size(ps_coords, 2)
        res[:, i] .= __furl_moms(view(ps_coords, :, i))
    end
    return res
end

function _furl_moms(moms::NTuple{N,Float64}) where {N}
    return Tuple(_furl_moms(Vector{Float64}([moms...])))
end

function _rnd_boost_params(rng)
    xyz = Tuple(rand(rng, 3))
    xyz = @. (2 * xyz - 1) / sqrt(3)
    return xyz
end

function _generate_onshell_two_body_moms(rng, masses, sqrt_s)
    rnd_boost = Boost(BetaVector(_rnd_boost_params(rng)...))
    p1_restframe = SFourMomentum(masses[1], 0, 0, 0)

    E2_restframe = (sqrt_s^2 - sum(masses .^ 2)) / (2 * masses[1])
    rho2 = sqrt(E2_restframe^2 - masses[2]^2)
    cth2 = 2 * rand(rng) - one(sqrt_s)
    sth2 = sqrt(one(cth2) - cth2^2)

    phi2 = 2 * pi * rand(rng)
    sphi2, cphi2 = sincos(phi2)

    p2_restframe = SFourMomentum(
        E2_restframe, rho2 * sth2 * cphi2, rho2 * sth2 * sphi2, rho2 * cth2
    )

    return rnd_boost.((p1_restframe, p2_restframe))
end
