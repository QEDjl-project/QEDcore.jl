# rest system
abstract type AbstractTwoBodyRestSystem <: AbstractTwoBodyInPhaseSpaceLayout end

phase_space_dimension(proc, model, ::AbstractTwoBodyRestSystem) = 0

### Two body target system

"""
    TwoBodyRestSystem(coord::AbstractUnivariateCoordinates)

    Two body phase space layout for the system, where one of the incoming particles
    is assumed to be at rest.
"""
struct TwoBodyRestSystem{RESTIDX,COORD<:AbstractUnivariateCoordinates} <:
       AbstractInPhaseSpaceLayout
    coord::COORD

    function TwoBodyRestSystem{RESTIDX}(
        coord_name::COORD
    ) where {RESTIDX,COORD<:AbstractUnivariateCoordinates}
        # TODO: add validity check
        # - only particle idx != RESTIDX are allowed
        # - only Energy, CMSEnergy, Rapidity and SpatialMagnitude are allowed
        return new{RESTIDX,COORD}(coord_name)
    end
end

@inline TwoBodyRestSystem(::Val{IDX}, coord_name) where {IDX} =
    TwoBodyRestSystem{IDX}(coord_name)
@inline TwoBodyRestSystem(idx::Int, coord_name) = TwoBodyRestSystem(Val(idx), coord_name)
TwoBodyRestSystem() = TwoBodyRestSystem(1, Energy(2))
const TwoBodyTargetSystem{COORD} =
    TwoBodyRestSystem{1,COORD} where {COORD<:AbstractUnivariateCoordinates}
TwoBodyTargetSystem() = TwoBodyTargetSystem(Energy(2))
const TwoBodyBeamSystem{COORD} =
    TwoBodyRestSystem{2,COORD} where {COORD<:AbstractUnivariateCoordinates}
TwoBodyBeamSystem() = TwoBodyBeamSystem(Energy(1))

_select_moms(::Val{1}, ::Val{2}, P_rest, P_run) = (P_rest, P_run)
_select_moms(::Val{2}, ::Val{1}, P_rest, P_run) = (P_run, P_rest)
function _select_moms(rest_idx::Int, run_idx::Int, P_rest, P_run)
    return _select_moms(Val(rest_idx), Val(run_idx), P_rest, P_run)
end

function QEDcore._build_momenta(
    proc::AbstractProcessDefinition,
    ::AbstractPerturbativeModel,
    ::TwoBodyRestSystem{RESTIDX,<:Energy{RUNIDX}},
    in_coords,
) where {RESTIDX,RUNIDX}
    masses = mass.(incoming_particles(proc))
    mass_rest = masses[RESTIDX]
    P_rest = SFourMomentum(mass_rest, 0, 0, 0)

    mass_run = masses[RUNIDX]
    energy_run = @inbounds in_coords[1]

    rho_run = sqrt(energy_run^2 - mass_run^2)
    P_run = SFourMomentum(energy_run, 0, 0, rho_run)

    return _select_moms(RESTIDX, RUNIDX, P_rest, P_run)
end

function _build_momenta(
    proc::AbstractProcessDefinition,
    ::AbstractPerturbativeModel,
    ::TwoBodyRestSystem{RESTIDX,<:SpatialMagnitude{RUNIDX}},
    in_coords,
) where {RESTIDX,RUNIDX}
    masses = mass.(incoming_particles(proc))
    mass_rest = masses[RESTIDX]
    P_rest = SFourMomentum(mass_rest, 0, 0, 0)

    mass_run = masses[RUNIDX]
    rho_run = @inbounds in_coords[1]

    energy_run = sqrt(rho_run^2 + mass_run^2)
    P_run = SFourMomentum(energy_run, 0, 0, rho_run)

    return _select_moms(RESTIDX, RUNIDX, P_rest, P_run)
end

_the_other(::Val{1}) = 2
_the_other(::Val{2}) = 1
_the_other(i::Int) = _the_other(Val(i))

function _build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractPerturbativeModel,
    in_psl::TwoBodyRestSystem{RESTIDX,<:CMSEnergy},
    in_coords,
) where {RESTIDX}
    RUNIDX = _the_other(RESTIDX)

    masses = mass.(incoming_particles(proc))
    mass_rest = @inbounds masses[RESTIDX]
    P_rest = SFourMomentum(mass_rest, 0, 0, 0)

    mass_run = @inbounds masses[RUNIDX]
    sqrt_s_run = @inbounds in_coords[1]

    energy_run = (sqrt_s_run^2 - mass_run^2 - mass_rest^2) / (2 * mass_rest)
    rho_run = sqrt(energy_run^2 - mass_run^2)

    P_run = SFourMomentum(energy_run, 0, 0, rho_run)

    return _select_moms(RESTIDX, RUNIDX, P_rest, P_run)
end

function QEDcore._build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractPerturbativeModel,
    in_psl::TwoBodyRestSystem{RESTIDX,<:Rapidity{RUNIDX}},
    in_coords,
) where {RESTIDX,RUNIDX}
    masses = mass.(incoming_particles(proc))
    mass_rest = @inbounds masses[RESTIDX]
    P_rest = SFourMomentum(mass_rest, 0, 0, 0)

    mass_run = @inbounds masses[RUNIDX]
    rapidity_run = @inbounds in_coords[1]
    energy_run = cosh(rapidity_run)
    rho_run = sinh(rapidity_run)
    P_run = SFourMomentum(energy_run, 0, 0, rho_run)

    return _select_moms(RESTIDX, RUNIDX, P_rest, P_run)
end
