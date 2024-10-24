# heads on systems

abstract type AbstractTwoBodyHeadsOnSystem <: AbstractTwoBodyInPhaseSpaceLayout end

struct TwoBodyCenterOfMomentumSystem{COORD} <: AbstractTwoBodyHeadsOnSystem
    coord::COORD
end
TwoBodyCenterOfMomentumSystem() = TwoBodyCenterOfMomentumSystem(Energy(1))

phase_space_dimension(::TwoBodyCenterOfMomentumSystem) = 1

function _build_momenta(
    proc::AbstractProcessDefinition,
    model::PerturbativeQED,
    in_psl::TwoBodyCenterOfMomentumSystem{<:Energy{1}},
    in_coords,
)
    mass1, mass2 = mass.(incoming_particles(proc))

    energy1 = @inbounds in_coords[1]

    rho1 = sqrt(energy1^2 - mass1^2)
    P1 = SFourMomentum(energy1, 0, 0, rho1)

    energy2 = sqrt(energy1^2 - mass1^2 + mass2^2)
    rho2 = rho1
    P2 = SFourMomentum(energy2, 0, 0, -rho2)
    return P1, P2
end

function _build_momenta(
    proc::AbstractProcessDefinition,
    model::PerturbativeQED,
    in_psl::TwoBodyCenterOfMomentumSystem{<:Energy{2}},
    in_coords,
)
    mass1, mass2 = mass.(incoming_particles(proc))

    energy2 = @inbounds in_coords[1]

    rho2 = sqrt(energy2^2 - mass2^2)
    P2 = SFourMomentum(energy2, 0, 0, -rho2)

    energy1 = sqrt(energy2^2 - mass2^2 + mass1^2)
    rho1 = rho2
    P1 = SFourMomentum(energy1, 0, 0, rho1)
    return P1, P2
end
