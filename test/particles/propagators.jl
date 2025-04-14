using Random
using QEDcore

RNG = MersenneTwister(137137)

function _rand_momentum(rng::AbstractRNG, ::Type{T}) where {T <: Number}
    return SFourMomentum(rand(rng, T, 4))
end

groundtruth_propagator(::Photon, mom) = one(eltype(mom)) / (mom * mom)
function groundtruth_propagator(particle::FermionLike, mom)
    return (slashed(mom) + mass(eltype(mom), particle) * one(DiracMatrix)) /
        (mom * mom - mass(eltype(mom), particle)^2)
end

@testset "propagators with $FLOAT_T" for FLOAT_T in (Float16, Float32, Float64)
    @testset "$P" for P in (Electron(), Positron(), Photon())
        mom = _rand_momentum(RNG, FLOAT_T)
        groundtruth = groundtruth_propagator(P, mom)
        test_prop = propagator(P, mom)
        @test isapprox(test_prop, groundtruth, atol = zero(FLOAT_T), rtol = sqrt(eps(FLOAT_T)))
        if P isa Photon
            @test test_prop isa FLOAT_T
        else
            @test test_prop isa DiracMatrix{Complex{FLOAT_T}}
        end
    end
end
