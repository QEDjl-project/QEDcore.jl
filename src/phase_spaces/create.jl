"""
    ParticleStateful{DIR, SPECIES}(mom::AbstractFourMomentum)
    ParticleStateful{DIR, SPECIES, EL}(mom::EL)

Construct a [`ParticleStateful`](@ref) from the given momentum on a fully or partially specified type.
"""
@inline function ParticleStateful{DIR, SPECIES}(
        mom::AbstractFourMomentum
    ) where {DIR <: ParticleDirection, SPECIES <: AbstractParticleType}
    return ParticleStateful(DIR(), SPECIES(), mom)
end

@inline function ParticleStateful{DIR, SPECIES, EL}(
        mom::EL
    ) where {DIR <: ParticleDirection, SPECIES <: AbstractParticleType, EL <: AbstractFourMomentum}
    return ParticleStateful(DIR(), SPECIES(), mom)
end

# PSP constructors from particle statefuls
"""
    InPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_ps::Tuple{ParticleStateful},
    )

    Construct a [`PhaseSpacePoint`](@ref) with only input particles from [`ParticleStateful`](@ref)s. The result will be `<: InPhaseSpacePoint` but **not** `<: OutPhaseSpacePoint`.
"""
function InPhaseSpacePoint(
        proc::PROC, model::MODEL, psl::PSL, in_ps::IN_PARTICLES
    ) where {
        PROC <: AbstractProcessDefinition,
        MODEL <: AbstractModelDefinition,
        PSL <: AbstractPhaseSpaceLayout,
        IN_PARTICLES <: Tuple{Vararg{ParticleStateful}},
    }
    return PhaseSpacePoint(proc, model, psl, in_ps, ())
end

"""
    OutPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        out_ps::Tuple{ParticleStateful},
    )

Construct a [`PhaseSpacePoint`](@ref) with only output particles from [`ParticleStateful`](@ref)s. The result will be `<: OutPhaseSpacePoint` but **not** `<: InPhaseSpacePoint`.
"""
function OutPhaseSpacePoint(
        proc::PROC, model::MODEL, psl::PSL, out_ps::OUT_PARTICLES
    ) where {
        PROC <: AbstractProcessDefinition,
        MODEL <: AbstractModelDefinition,
        PSL <: AbstractPhaseSpaceLayout,
        OUT_PARTICLES <: Tuple{Vararg{ParticleStateful}},
    }
    return PhaseSpacePoint(proc, model, psl, (), out_ps)
end

# PSP constructors from momenta

"""
    PhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_momenta::NTuple{N,AbstractFourMomentum},
        out_momenta::NTuple{M,AbstractFourMomentum},
    )

Construct the phase space point from given momenta of incoming and outgoing particles regarding a given process.
"""
function PhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_momenta::NTuple{N, ELEMENT},
        out_momenta::NTuple{M, ELEMENT},
    ) where {N, M, ELEMENT <: AbstractFourMomentum}
    in_particles = _build_particle_statefuls(proc, in_momenta, Incoming())
    out_particles = _build_particle_statefuls(proc, out_momenta, Outgoing())

    return PhaseSpacePoint(proc, model, psl, in_particles, out_particles)
end

"""
    InPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_momenta::NTuple{N,AbstractFourMomentum},
    )

Construct a [`PhaseSpacePoint`](@ref) with only input particles from given momenta. The result will be `<: InPhaseSpacePoint` but **not** `<: OutPhaseSpacePoint`.
"""
function InPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_momenta::NTuple{N, ELEMENT},
    ) where {N, ELEMENT <: AbstractFourMomentum}
    in_particles = _build_particle_statefuls(proc, in_momenta, Incoming())

    return PhaseSpacePoint(proc, model, psl, in_particles, ())
end

"""
    OutPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        out_momenta::NTuple{N,AbstractFourMomentum},
    )

Construct a [`PhaseSpacePoint`](@ref) with only output particles from given momenta. The result will be `<: OutPhaseSpacePoint` but **not** `<: InPhaseSpacePoint`.
"""
function OutPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        out_momenta::NTuple{N, ELEMENT},
    ) where {N, ELEMENT <: AbstractFourMomentum}
    out_particles = _build_particle_statefuls(proc, out_momenta, Outgoing())

    return PhaseSpacePoint(proc, model, psl, (), out_particles)
end

# PSP constructors from coordinates

"""
    PhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_coords::NTuple{N,Real},
        out_coords::NTuple{M,Real},
    )

Construct a [`PhaseSpacePoint`](@ref) from given coordinates by using the `_generate_momenta` interface.
Alternative versions of this constructor exist using `SVector`s instead of `NTuple`s, and a version where only one `NTuple` or one `SVector` is given containing all coordinates.
"""
function PhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_coords::NTuple{N, Real},
        out_coords::NTuple{M, Real},
    ) where {N, M}
    in_dim = phase_space_dimension(proc, model, in_phase_space_layout(psl))

    in_dim == length(in_coords) || throw(
        ArgumentError(
            "the in-phase-space dimension must match the number of in-coordinates"
        )
    )
    out_dim = phase_space_dimension(proc, model, psl)

    out_dim == length(out_coords) || throw(
        ArgumentError(
            "the out-phase-space dimension must match the number of out-coordinates"
        )
    )
    in_ps, out_ps = QEDbase._build_momenta(proc, model, psl, in_coords, out_coords)
    return PhaseSpacePoint(proc, model, psl, in_ps, out_ps)
end
function PhaseSpacePoint(
        p::AbstractProcessDefinition,
        m::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        coords::Tuple,
    )

    in_dim = phase_space_dimension(p, m, in_phase_space_layout(psl))
    out_dim = phase_space_dimension(p, m, psl)

    in_dim + out_dim == length(coords) || throw(
        ArgumentError(
            "sum of in- and out-phase-space dimension must match the number of coordinates"
        )
    )
    return PhaseSpacePoint(
        p,
        m,
        psl,
        ntuple(i -> coords[i], in_dim),
        ntuple(i -> coords[in_dim + i], out_dim),
    )
end

function PhaseSpacePoint(
        p::AbstractProcessDefinition,
        m::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        coords::SVector,
    )
    return PhaseSpacePoint(p, m, psl, Tuple(coords))
end

function PhaseSpacePoint(
        p::AbstractProcessDefinition,
        m::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_coords::SVector,
        out_coords::SVector,
    )
    return PhaseSpacePoint(p, m, psl, Tuple(in_coords), Tuple(out_coords))
end

"""
    InPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractPhaseSpaceLayout,
        in_coords::NTuple{N,Real},
    )

Construct a [`PhaseSpacePoint`](@ref) from given coordinates by using the `_generate_momenta` interface. The result will be `<: InPhaseSpacePoint` but **not** `<: OutPhaseSpacePoint`.

!!! note
    A similar function for [`OutPhaseSpacePoint`](@ref) does not exist from coordinates, only a full [`PhaseSpacePoint`](@ref).
"""
function InPhaseSpacePoint(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractInPhaseSpaceLayout,
        in_coords::NTuple{N, Real},
    ) where {N}
    in_ps = QEDbase._build_momenta(proc, model, psl, in_coords)
    return InPhaseSpacePoint(proc, model, psl, in_ps)
end
