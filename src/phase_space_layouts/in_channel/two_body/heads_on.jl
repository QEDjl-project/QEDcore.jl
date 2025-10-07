"""

    AbstractAxis

Abstract base type to describe the axis of a vector, e.g. the k-vector of a photon. Mostly used for multiple dispatch.
"""
abstract type AbstractAxis end
abstract type AbstractDefiniteAxis <: AbstractAxis end
abstract type AbstractIndefiniteAxis <: AbstractAxis end

struct XAxis <: AbstractDefiniteAxis end
struct YAxis <: AbstractDefiniteAxis end
struct ZAxis <: AbstractDefiniteAxis end

abstract type AbstractTwoBodyHeadsOnSystem <: AbstractTwoBodyInPhaseSpaceLayout end
QEDbase.phase_space_dimension(proc, model, ::AbstractTwoBodyHeadsOnSystem) = 2 # E1,E2

"""
    HeadsOnSystem([dir::AbstractDefiniteAxis], particle1::AbstractParticle, particle2::AbstractParticle)

Define a general head-on two-body system where `particle1` and `particle2`
counter-propagate along the given definite axis `dir`
(defaults to the `ZAxis`).
"""
struct HeadsOnSystem{D <: AbstractDefiniteAxis, P1 <: AbstractParticle, P2 <: AbstractParticle} <: AbstractTwoBodyHeadsOnSystem
    dir::D
    particle1::P1
    particle2::P2
end

QEDbase.particle_direction(psl::HeadsOnSystem) = psl.dir
HeadsOnSystem(p1::AbstractParticle, p2::AbstractParticle) = HeadsOnSystem(ZAxis(), p1, p2)

@inline _build_directed_moms(dir::XAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, -rho1, 0, 0), SFourMomentum(E2, rho2, 0, 0))
@inline _build_directed_moms(dir::YAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, 0, -rho1, 0), SFourMomentum(E2, 0, rho2, 0))
@inline _build_directed_moms(dir::ZAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, 0, 0, -rho1), SFourMomentum(E2, 0, 0, rho2))

function QEDbase._build_momenta(
        ::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::HeadsOnSystem,
        in_coords::NTuple{2, T}
    ) where {T <: Real}
    @inbounds E1, E2 = in_coords
    rho1 = sqrt((E1 - mass(psl.particle1)) * (E1 + mass(psl.particle1)))
    rho2 = sqrt((E2 - mass(psl.particle2)) * (E2 + mass(psl.particle2)))

    return _build_directed_moms(particle_direction(psl), E1, rho1, E1, rho2)
end

### photon-electron system
"""
    PhotonElectronHeadsOnSystem([dir::AbstractDefiniteAxis])

Define a head-on collision system between a photon and an electron,
where both particles propagate along the given definite axis `dir`
(defaults to the `ZAxis`).
"""
const PhotonElectronHeadsOnSystem{D <: AbstractDefiniteAxis} = HeadsOnSystem{D, Electron, Photon}
PhotonElectronHeadsOnSystem(dir::AbstractDefiniteAxis) = HeadsOnSystem(dir, Electron(), Photon())
PhotonElectronHeadsOnSystem() = PhotonElectronHeadsOnSystem(ZAxis())

@inline _build_directed_moms(dir::XAxis, E, rho, om) = (SFourMomentum(E, -rho, 0, 0), SFourMomentum(om, om, 0, 0))
@inline _build_directed_moms(dir::YAxis, E, rho, om) = (SFourMomentum(E, 0, -rho, 0), SFourMomentum(om, 0, om, 0))
@inline _build_directed_moms(dir::ZAxis, E, rho, om) = (SFourMomentum(E, 0, 0, -rho), SFourMomentum(om, 0, 0, om))

function QEDbase._build_momenta(
        ::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::PhotonElectronHeadsOnSystem,
        in_coords::NTuple{2, T}
    ) where {T <: Real}
    @inbounds E, om = in_coords
    rho = sqrt((E - mass(Electron())) * (E + mass(Electron())))

    return _build_directed_moms(particle_direction(psl), E, rho, om)
end
