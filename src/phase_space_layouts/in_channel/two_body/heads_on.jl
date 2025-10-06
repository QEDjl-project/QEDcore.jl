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
QEDbase.phase_space_dimension(proc, model, ::AbstractTwoBodyHeadsOnSystem) = 2 # E, omega

"""
    PhotonElectronHeadsOnSystem([dir::AbstractDefiniteAxis])

Define a head-on collision system between a photon and an electron,
where both particles propagate along the given definite axis `dir`
(defaults to the `ZAxis`).
"""
struct PhotonElectronHeadsOnSystem{D <: AbstractDefiniteAxis} <: AbstractTwoBodyHeadsOnSystem
    dir::D
end
QEDbase.particle_direction(psl::PhotonElectronHeadsOnSystem) = psl.dir
PhotonElectronHeadsOnSystem() = PhotonElectronHeadsOnSystem(ZAxis())

#Base.broadcastable(psl::PhotonElectronHeadsOnSystem) = Ref(psl)

@inline _build_directed_moms(dir::XAxis, E, rho, om) = (SFourMomentum(E, -rho, 0, 0), SFourMomentum(om, om, 0, 0))
@inline _build_directed_moms(dir::YAxis, E, rho, om) = (SFourMomentum(E, 0, -rho, 0), SFourMomentum(om, 0, om, 0))
@inline _build_directed_moms(dir::ZAxis, E, rho, om) = (SFourMomentum(E, 0, 0, -rho), SFourMomentum(om, 0, 0, om))

function QEDbase._build_momenta(
        ::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::PhotonElectronHeadsOnSystem,
        in_coords::NTuple{2, T}
    ) where {T <: Real}
    E, om = in_coords
    rho = sqrt(E^2 - one(E))

    return _build_directed_moms(particle_direction(psl), E, rho, om)
end
