@inline function _booster_fermion(mom::AbstractFourMomentum{T}, mass::Real) where {T <: Real}
    return (slashed(mom) + mass * one(DiracMatrix{T})) / (sqrt(abs(getT(mom)) + mass))
end

@inline function _booster_antifermion(
        mom::AbstractFourMomentum{T}, mass::Real
    ) where {T <: Real}
    return (mass * one(DiracMatrix{T}) - slashed(mom)) / (sqrt(abs(getT(mom)) + mass))
end

function QEDbase.base_state(
        particle::Fermion, ::Incoming, mom::AbstractFourMomentum{T}, spin::AbstractDefiniteSpin
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_fermion(mom, mass(T, particle))
    return BiSpinor{T_COMPLEX}(@inbounds booster[:, QEDbase._spin_index(spin)])
end

function QEDbase.base_state(
        particle::Fermion, ::Incoming, mom::AbstractFourMomentum{T}, spin::AllSpin
    ) where {T <: Real}
    booster = _booster_fermion(mom, mass(T, particle))
    return SVector(BiSpinor(@inbounds booster[:, 1]), BiSpinor(@inbounds booster[:, 2]))
end

function QEDbase.base_state(
        particle::AntiFermion,
        ::Incoming,
        mom::AbstractFourMomentum{T},
        spin::AbstractDefiniteSpin,
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_antifermion(mom, mass(T, particle))
    return AdjointBiSpinor(BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin) + 2])) *
        (@inbounds gamma(T_COMPLEX)[1])
end

function QEDbase.base_state(
        particle::AntiFermion, ::Incoming, mom::AbstractFourMomentum{T}, spin::AllSpin
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_antifermion(mom, mass(T, particle))
    return SVector(
        AdjointBiSpinor(@inbounds BiSpinor(booster[:, 3])) *
            (@inbounds gamma(T_COMPLEX)[1]),
        AdjointBiSpinor(@inbounds BiSpinor(booster[:, 4])) *
            (@inbounds gamma(T_COMPLEX)[1]),
    )
end

function QEDbase.base_state(
        particle::Fermion, ::Outgoing, mom::AbstractFourMomentum{T}, spin::AbstractDefiniteSpin
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_fermion(mom, mass(T, particle))
    return AdjointBiSpinor(BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin)])) *
        (@inbounds gamma(T_COMPLEX)[1])
end

function QEDbase.base_state(
        particle::Fermion, ::Outgoing, mom::AbstractFourMomentum{T}, spin::AllSpin
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_fermion(mom, mass(T, particle))
    return SVector(
        AdjointBiSpinor(BiSpinor(@inbounds booster[:, 1])) *
            (@inbounds gamma(T_COMPLEX)[1]),
        AdjointBiSpinor(BiSpinor(@inbounds booster[:, 2])) *
            (@inbounds gamma(T_COMPLEX)[1]),
    )
end

function QEDbase.base_state(
        particle::AntiFermion,
        ::Outgoing,
        mom::AbstractFourMomentum{T},
        spin::AbstractDefiniteSpin,
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_antifermion(mom, mass(T, particle))
    return BiSpinor{T_COMPLEX}(@inbounds booster[:, QEDbase._spin_index(spin) + 2])
end

function QEDbase.base_state(
        particle::AntiFermion, ::Outgoing, mom::AbstractFourMomentum{T}, spin::AllSpin
    ) where {T <: Real}
    T_COMPLEX = _complex_from_real_t(T)
    booster = _booster_antifermion(mom, mass(T, particle))
    return SVector(
        BiSpinor{T_COMPLEX}(@inbounds booster[:, 3]),
        BiSpinor{T_COMPLEX}(@inbounds booster[:, 4]),
    )
end

function _photon_state(pol::AllPolarization, mom::AbstractFourMomentum{T}) where {T <: Real}
    cth = getCosTheta(mom)
    sth = sq_diff_sqrt(1, cth)
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SVector(
        SLorentzVector{T}(0.0, cth * cos_phi, cth * sin_phi, -sth),
        SLorentzVector{T}(0.0, -sin_phi, cos_phi, 0.0),
    )
end

function _photon_state(pol::PolarizationX, mom::AbstractFourMomentum{T}) where {T <: Real}
    cth = getCosTheta(mom)
    sth = sq_diff_sqrt(1, cth)
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SLorentzVector{T}(0.0, cth * cos_phi, cth * sin_phi, -sth)
end

function _photon_state(pol::PolarizationY, mom::AbstractFourMomentum{T}) where {T <: Real}
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SLorentzVector{T}(0.0, -sin_phi, cos_phi, 0.0)
end

@inline function QEDbase.base_state(
        particle::Photon, ::ParticleDirection, mom::AbstractFourMomentum, pol::AllPolarization
    )
    return _photon_state(pol, mom)
end

@inline function QEDbase.base_state(
        particle::Photon,
        ::ParticleDirection,
        mom::AbstractFourMomentum,
        pol::AbstractPolarization,
    )
    return _photon_state(pol, mom)
end
