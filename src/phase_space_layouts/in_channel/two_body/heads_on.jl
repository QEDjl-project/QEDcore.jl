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

@inline _build_directed_moms(dir::XAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, -rho1, 0, 0), SFourMomentum(E2, rho2, 0, 0))
@inline _build_directed_moms(dir::YAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, 0, -rho1, 0), SFourMomentum(E2, 0, rho2, 0))
@inline _build_directed_moms(dir::ZAxis, E1, rho1, E2, rho2) = (SFourMomentum(E1, 0, 0, -rho1), SFourMomentum(E2, 0, 0, rho2))

abstract type AbstractTwoBodyHeadsOnSystem <: AbstractTwoBodyInPhaseSpaceLayout end
QEDbase.phase_space_dimension(proc, model, ::AbstractTwoBodyHeadsOnSystem) = 2 # E1,E2

"""
    HeadsOnSystem([dir::AbstractDefiniteAxis])

Define a general head-on two-body system where `particle1` and `particle2`
counter-propagate along the given definite axis `dir`
(defaults to the `ZAxis`).
"""
struct HeadsOnSystem{D <: AbstractDefiniteAxis} <: AbstractTwoBodyHeadsOnSystem
    dir::D
end

QEDbase.particle_direction(psl::HeadsOnSystem) = psl.dir
HeadsOnSystem() = HeadsOnSystem(ZAxis())


function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::HeadsOnSystem,
        in_coords::NTuple{2, T}
    ) where {T <: Real}
    @inbounds E1, E2 = in_coords
    part1, part2 = incoming_particles(proc)
    rho1 = _mag_from_E(E1, mass(part1))
    rho2 = _mag_from_E(E2, mass(part2))

    return _build_directed_moms(particle_direction(psl), E1, rho1, E2, rho2)
end

### Center-of-momentum system

abstract type AbstractCenterOfMomentumSystem <: AbstractTwoBodyHeadsOnSystem end
"""
    CenterOfMomentumSystem([coord::AbstractUnivariateCoordinate=CMSEnergy()],[dir::AbstractDefiniteAxis=ZAxis()])

Define a two-body phase-space layout representing the center-of-momentum system (CMS),
where the total spatial momentum vanishes and both particles counter-propagate
along the definite axis `dir` (defaults to `ZAxis()`).

The system can be parameterized either by the total center-of-mass energy (`CMSEnergy()`)
or by the energy of one of the two particles (`Energy{1}()` or `Energy{2}()`),
given via the `coord` argument.

# Examples
```julia
# Parameterize by total CoM energy (default)
psl = CenterOfMomentumSystem(CMSEnergy(), ZAxis())
psl = CenterOfMomentumSystem(CMSEnergy())           # same as above

# Parameterize by the energy of the first or second particle
psl1 = CenterOfMomentumSystem(Energy{1}())
psl1 = CenterOfMomentumSystem(Energy{2}())

# Specify a different propagation axis
psl_x = CenterOfMomentumSystem(CMSEnergy(), XAxis())
```
"""
struct CenterOfMomentumSystem{COORD, D} <: AbstractCenterOfMomentumSystem
    coord::COORD
    dir::D

    function CenterOfMomentumSystem(
            coord::CMSEnergy,
            dir::D,
        ) where {
            D <: AbstractDefiniteAxis,
        }
        return new{CMSEnergy, D}(coord)
    end

    function CenterOfMomentumSystem(
            coord::COORD,
            dir::D,
        ) where {
            PIDX,
            COORD <: AbstractSingleParticleCoordinate{PIDX},
            D <: AbstractDefiniteAxis,
        }

        # TODO: is it fine to check it like this?
        PIDX in (1, 2) || throw(
            ArgumentError(
                "coordinate for Center-of-momentum system must be for the first or second particle"
            )
        )

        return new{COORD, D}(coord, dir)
    end
end

CenterOfMomentumSystem(dir::AbstractDefiniteAxis) = CenterOfMomentumSystem(CMSEnergy(), dir)
CenterOfMomentumSystem(coord::AbstractUnivariateCoordinate) = CenterOfMomentumSystem(coord, ZAxis())
CenterOfMomentumSystem() = CenterOfMomentumSystem(CMSEnergy(), ZAxis())

QEDbase.phase_space_dimension(proc, model, ::CenterOfMomentumSystem) = 1 # sqrt_s, E1, or E2
QEDbase.particle_direction(psl::CenterOfMomentumSystem) = psl.dir

function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::CenterOfMomentumSystem{CMSEnergy},
        in_coords::NTuple{1, T}
    ) where {T <: Real}
    @inbounds sqrt_s = in_coords[1]
    part1, part2 = incoming_particles(proc)
    s = sqrt_s^2
    m1 = mass(part1)
    m2 = mass(part2)

    E1 = (s + (m1 - m2) * (m1 + m2)) / (2 * sqrt_s)
    E2 = sqrt_s - E1

    rho1 = _mag_from_E(E1, m1)
    rho2 = _mag_from_E(E2, m2)

    return _build_directed_moms(particle_direction(psl), E1, rho1, E2, rho2)
end

# TODO: does the indexing causes overhead?
function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        psl::CenterOfMomentumSystem{Energy{IDX}},
        in_coords::NTuple{1, T}
    ) where {IDX, T <: Real}
    @inbounds E = in_coords[1]
    IDX_other = _the_other(IDX)

    parts = incoming_particles(proc)
    part = parts[IDX]
    part_other = parts[IDX_other]

    m = mass(part)
    m_other = mass(part_other)

    E_other = sqrt((E - m) * (E + m) + m_other^2)

    rho = _mag_from_E(E, m)
    rho_other = _mag_from_E(E_other, m_other)

    P, P_other = _build_directed_moms(particle_direction(psl), E, rho, E_other, rho_other)
    return _order_moms(IDX, IDX_other, P, P_other)
end
