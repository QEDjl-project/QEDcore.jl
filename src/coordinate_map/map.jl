abstract type AbstractCoordianteMap end

struct CoordinateMap{P,M,PSL<:AbstractPhaseSpaceLayout} <: AbstractCoordianteMap
    proc::P
    model::M
    psl::PSL

    # TODO: validity check
    # - compatibility of proc, model, psl
end

# make the transform callable
@inline function (coord_map::CoordinateMap{P,M,PSL})(
    in_coords::Tuple
) where {P,M,PSL<:AbstractInPhaseSpaceLayout}
    return build_momenta(coord_map.proc, coord_map.model, coord_map.psl, in_coords)
end

# make the transform callable
@inline function (coord_map::CoordinateMap{P,M,PSL})(
    in_coords::Tuple, out_coords::Tuple
) where {P,M,PSL<:AbstractOutPhaseSpaceLayout}
    in_moms = build_momenta(
        coord_map.proc, coord_map.model, in_phase_space_layout(coord_map.psl), in_coords
    )
    Ptot = sum(in_moms)
    return in_moms,
    build_momenta(coord_map.proc, coord_map.model, Ptot, coord_map.psl, out_coords)
end
