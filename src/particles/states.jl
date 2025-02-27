@inline function _booster_fermion(mom::AbstractFourMomentum, mass::Real)
    return (slashed(mom) + mass * one(DiracMatrix)) / (sqrt(abs(getT(mom)) + mass))
end

@inline function _booster_antifermion(mom::AbstractFourMomentum, mass::Real)
    return (mass * one(DiracMatrix) - slashed(mom)) / (sqrt(abs(getT(mom)) + mass))
end

function QEDbase.base_state(
    particle::Fermion, ::Incoming, mom::AbstractFourMomentum, spin::AbstractDefiniteSpin
)
    booster = _booster_fermion(mom, mass(particle))
    return BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin)])
end

function QEDbase.base_state(
    particle::Fermion, ::Incoming, mom::AbstractFourMomentum, spin::AllSpin
)
    booster = _booster_fermion(mom, mass(particle))
    return SVector(BiSpinor(@inbounds booster[:, 1]), BiSpinor(@inbounds booster[:, 2]))
end

function QEDbase.base_state(
    particle::AntiFermion, ::Incoming, mom::AbstractFourMomentum, spin::AbstractDefiniteSpin
)
    booster = _booster_antifermion(mom, mass(particle))
    return AdjointBiSpinor(BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin) + 2])) *
           (@inbounds GAMMA[1])
end

function QEDbase.base_state(
    particle::AntiFermion, ::Incoming, mom::AbstractFourMomentum, spin::AllSpin
)
    booster = _booster_antifermion(mom, mass(particle))
    return SVector(
        AdjointBiSpinor(@inbounds BiSpinor(booster[:, 3])) * (@inbounds GAMMA[1]),
        AdjointBiSpinor(@inbounds BiSpinor(booster[:, 4])) * (@inbounds GAMMA[1]),
    )
end

function QEDbase.base_state(
    particle::Fermion, ::Outgoing, mom::AbstractFourMomentum, spin::AbstractDefiniteSpin
)
    booster = _booster_fermion(mom, mass(particle))
    return AdjointBiSpinor(BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin)])) *
           (@inbounds GAMMA[1])
end

function QEDbase.base_state(
    particle::Fermion, ::Outgoing, mom::AbstractFourMomentum, spin::AllSpin
)
    booster = _booster_fermion(mom, mass(particle))
    return SVector(
        AdjointBiSpinor(BiSpinor(@inbounds booster[:, 1])) * (@inbounds GAMMA[1]),
        AdjointBiSpinor(BiSpinor(@inbounds booster[:, 2])) * (@inbounds GAMMA[1]),
    )
end

function QEDbase.base_state(
    particle::AntiFermion, ::Outgoing, mom::AbstractFourMomentum, spin::AbstractDefiniteSpin
)
    booster = _booster_antifermion(mom, mass(particle))
    return BiSpinor(@inbounds booster[:, QEDbase._spin_index(spin) + 2])
end

function QEDbase.base_state(
    particle::AntiFermion, ::Outgoing, mom::AbstractFourMomentum, spin::AllSpin
)
    booster = _booster_antifermion(mom, mass(particle))
    return SVector(BiSpinor(@inbounds booster[:, 3]), BiSpinor(@inbounds booster[:, 4]))
end

function _photon_state(pol::AllPolarization, mom::AbstractFourMomentum)
    cth = getCosTheta(mom)
    sth = sqrt(1 - cth^2)
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SVector(
        SLorentzVector{Float64}(0.0, cth * cos_phi, cth * sin_phi, -sth),
        SLorentzVector{Float64}(0.0, -sin_phi, cos_phi, 0.0),
    )
end

function _photon_state(pol::PolarizationX, mom::AbstractFourMomentum)
    cth = getCosTheta(mom)
    sth = sqrt(1 - cth^2)
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SLorentzVector{Float64}(0.0, cth * cos_phi, cth * sin_phi, -sth)
end

function _photon_state(pol::PolarizationY, mom::AbstractFourMomentum)
    cos_phi = getCosPhi(mom)
    sin_phi = getSinPhi(mom)
    return SLorentzVector{Float64}(0.0, -sin_phi, cos_phi, 0.0)
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
