struct CoordinateMapCached{
    P<:AbstractProcessDefinition,M<:AbstractModelDefinition,PSL<:AbstractPhaseSpaceLayout,TM
} <: AbstractCoordianteMap
    proc::P
    model::M
    psl::PSL
    in_moms::TM

    # TODO: consider compat check for (proc,model,psl)
end

function CoordinateMapCached(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    psl::AbstractInPhaseSpaceLayout,
    in_coords::NTuple{N,T},
) where {N,T<:Real}
    in_moms = build_momenta(proc, model, psl, in_coords)
    return CoordinateMapCached(proc, model, psl, in_moms)
end

function CoordinateMapCached(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    psl::AbstractOutPhaseSpaceLayout,
    in_coords::NTuple{N,T},
) where {N,T<:Real}
    in_moms = build_momenta(proc, model, in_phase_space_layout(psl), in_coords)
    return CoordinateMapCached(proc, model, psl, in_moms)
end

# make the transform callable: for in_psl maps return the cached
@inline function (
    coord_map::CoordinateMapCached{P,M,PSL}
)() where {P,M,PSL<:AbstractInPhaseSpaceLayout}
    return getfield(coord_map, :in_moms)
end

# make the transform callable: for out_psl maps
@inline function (coord_map::CoordinateMapCached{P,M,PSL})(
    out_coords::Tuple
) where {P,M,PSL<:AbstractOutPhaseSpaceLayout}
    in_moms = coord_map.in_moms
    Ptot = sum(in_moms)
    return build_momenta(coord_map.proc, coord_map.model, Ptot, coord_map.psl, out_coords)
end
