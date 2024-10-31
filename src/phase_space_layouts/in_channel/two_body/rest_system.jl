"""
    AbstractTwoBodyRestSystem <: AbstractTwoBodyInPhaseSpaceLayout

An abstract type representing the phase space layout of a two-body system in which one particle
is at rest in the system's frame.

Concrete subtypes of `AbstractTwoBodyRestSystem` typically specify the coordinate system used
to parameterize the momentum of the non-resting particle, such as [`Energy`](@ref), [`SpatialMagnitude`](@ref),
[`CenterOfMomentumEnergy`](@ref), or [`Rapidity`](@ref). The choice of coordinate system impacts how the momenta of the
incoming particles are constructed.

This type serves as a base for defining custom layouts that rely on rest-frame assumptions
for two-body scattering or decay processes, and it facilitates method dispatch in momentum
construction functions like [`_build_momenta`](@ref).

# See Also
- [`TwoBodyRestSystem`](@ref): A concrete implementation of `AbstractTwoBodyRestSystem` that
    supports various univariate coordinates for describing the non-resting particle.
"""
abstract type AbstractTwoBodyRestSystem <: AbstractTwoBodyInPhaseSpaceLayout end

phase_space_dimension(proc, model, ::AbstractTwoBodyRestSystem) = 1

### Two body target system

"""
    TwoBodyRestSystem{RESTIDX, COORD<:AbstractUnivariateCoordinate}

Represents a two-body scattering system in the rest frame of one of the particles, where one particle (identified by `RESTIDX`) is at rest and the other particle's momentum is described by a coordinate system `COORD`. The system uses various univariate coordinates, such as energy, rapidity, or center-of-momentum energy, to parameterize the momentum of the moving particle.

This type allows for easy construction of the incoming particle momenta using the phase space layout for two-body systems, commonly used in high-energy physics processes. The system supports different coordinate systems for defining the moving particle's momenta, including energy-based and rapidity-based layouts.

# Fields
- `coord::COORD`: The coordinate type (`COORD`) used to describe the non-resting particle's momenta.

# Supported Coordinates
- `Energy`: Defines the energy of the moving particle.
- `SpatialMagnitude`: Defines the spatial momentum magnitude of the moving particle.
- `CMSEnergy`: Defines the total center-of-mass energy of the system.
- `Rapidity`: Defines the rapidity of the moving particle.

# Example Usages
The following examples show how to use different coordinate systems with `TwoBodyRestSystem` in conjunction with the `_build_momenta` interface.

### Example 1: Using `Energy` Coordinate

```Julia
psl = TwoBodyRestSystem(Energy(2))  # Particle 1 at rest, particle 2 described by its energy
in_coords = (100.0,)  # Energy of particle 2
momenta = build_momenta(proc, model, psl, in_coords)
```

This constructs the momenta assuming particle 1 is at rest, and particle 2 has an energy of 100 GeV.

### Example 2: Using `SpatialMagnitude` Coordinate

```Julia
psl = TwoBodyRestSystem(SpatialMagnitude(2))  # Particle 1 at rest, particle 2 described by its spatial magnitude
in_coords = (50.0,)  # Spatial momentum magnitude of particle 2
momenta = build_momenta(proc, model, psl, in_coords)
```

In this case, the moving particle's spatial momentum magnitude is given, and its energy is calculated based on its mass.

### Example 3: Using `CMSEnergy` Coordinate

```Julia
psl = TwoBodyRestSystem(1,CMSEnergy())  # Particle 1 at rest, particle 2 described by the center-of-mass energy
in_coords = (200.0,)  # Total center-of-mass energy of the system
momenta = build_momenta(proc, model, psl, in_coords)
```

Here, the center-of-mass energy of the system is provided, and the momenta of both particles are calculated while conserving energy and momentum.

### Example 4: Using `Rapidity` Coordinate

```Julia
psl = TwoBodyRestSystem(Rapidity(2))  # Particle 1 at rest, particle 2 described by its rapidity
in_coords = (1.0,)  # Rapidity of particle 2
momenta = _build_momenta(proc, model, psl, in_coords)
@assert isapprox(getMass(sum(momenta)),50.0)
```

In this case, the rapidity of particle 2 is given, and its energy and momentum are computed based on this parameter.

# Notes
- `RESTIDX` is the index of the particle that is at rest in the system.
- `COORD` specifies the coordinate type of the non-resting particle and must be a subtype of
    `AbstractUnivariateCoordinate`.
- The coordinate system should be compatible with the given particles and their masses,
    otherwise, an error may occur during momentum construction.

# Throws
- `ArgumentError` if invalid coordinates are used (e.g., unsupported coordinate types or
    incompatible indices).
"""
struct TwoBodyRestSystem{RESTIDX,COORD<:AbstractUnivariateCoordinate} <:
       AbstractTwoBodyRestSystem
    coord::COORD

    function TwoBodyRestSystem{RESTIDX}(
        coord_name::COORD
    ) where {RESTIDX,COORD<:AbstractSingleParticleCoordinate{RESTIDX}}
        throw(
            InvalidInputError(
                "the particle at rest and the particle parameterize by the coordinate must not have the same index",
            ),
        )
    end

    function TwoBodyRestSystem{RESTIDX}(
        coord_name::COORD
    ) where {RESTIDX,RUNIDX,COORD<:AbstractSingleParticleCoordinate{RUNIDX}}
        return new{RESTIDX,COORD}(coord_name)
    end

    function TwoBodyRestSystem{RESTIDX}(coord_name::CMSEnergy) where {RESTIDX}
        return new{RESTIDX,CMSEnergy}(coord_name)
    end
end

@inline TwoBodyRestSystem(::Val{RESTIDX}, coord_name) where {RESTIDX} =
    TwoBodyRestSystem{RESTIDX}(coord_name)
@inline TwoBodyRestSystem(rest_idx::Int, coord_name) =
    TwoBodyRestSystem(Val(rest_idx), coord_name)
@inline TwoBodyRestSystem(
    coord::COORD
) where {RUNIDX,COORD<:AbstractSingleParticleCoordinate{RUNIDX}} =
    TwoBodyRestSystem{_the_other(RUNIDX)}(coord)
TwoBodyRestSystem() = TwoBodyRestSystem(1, Energy(2))
const TwoBodyTargetSystem{COORD} =
    TwoBodyRestSystem{1,COORD} where {COORD<:AbstractUnivariateCoordinate}
TwoBodyTargetSystem() = TwoBodyTargetSystem(Energy(2))
const TwoBodyBeamSystem{COORD} =
    TwoBodyRestSystem{2,COORD} where {COORD<:AbstractUnivariateCoordinate}
TwoBodyBeamSystem() = TwoBodyBeamSystem(Energy(1))

function _build_momenta(
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

    return _order_moms(RESTIDX, RUNIDX, P_rest, P_run)
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

    return _order_moms(RESTIDX, RUNIDX, P_rest, P_run)
end

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

    return _order_moms(RESTIDX, RUNIDX, P_rest, P_run)
end

function _build_momenta(
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
    energy_run = mass_run * cosh(rapidity_run)
    rho_run = mass_run * sinh(rapidity_run)
    P_run = SFourMomentum(energy_run, 0, 0, rho_run)

    return _order_moms(RESTIDX, RUNIDX, P_rest, P_run)
end
