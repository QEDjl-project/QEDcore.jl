abstract type AbstractCoordinateMap end

"""
    CoordinateMap{P,M,PSL}(proc::P, model::M, psl::PSL)

A `CoordinateMap` represents a transformation that maps phase space coordinates to particle
momenta for a specific scattering process. This type encapsulates the scattering process
definition (`proc`), the physics model (`model`), and the phase space layout (`psl`).
The `CoordinateMap` is callable and supports the conversion of both incoming and outgoing
phase space coordinates into four-momenta.

## Fields
- `proc`: The scattering process definition, subtype of `AbstractProcessDefinition`.
- `model`: The physics model, subtype of `AbstractModelDefinition`.
- `psl`: The phase space layout, either incoming or outgoing, that maps coordinates to momenta.

## Usage

### Incoming Coordinates
The `CoordinateMap` can be called with a tuple of incoming phase space coordinates
(`in_coords`), which are used to compute the corresponding incoming particle four-momenta.
This is done by calling the phase space construction function [`QEDbase.build_momenta`](@extref) using
the `proc`, `model`, and `psl` provided in the `CoordinateMap`.

```julia
# Create a coordinate map for in phase space
in_coord_map = CoordinateMap(proc, model, in_psl)

# call on in-coordinates to build the momenta
in_coord_map(in_coords)
```

- **Arguments**:
    - `in_coords`: A tuple of phase space coordinates for the incoming particles.

- **Returns**: A tuple of four-momenta corresponding to the incoming particles.

### Incoming and Outgoing Coordinates
The `CoordinateMap` can also be called with both incoming (`in_coords`) and outgoing
(`out_coords`) phase space coordinates. The incoming momenta are computed first, and then
the total momentum (`Ptot`) is calculated by summing these momenta. This total momentum is
then used to compute the outgoing particle four-momenta using the provided outgoing phase
space layout.

```julia
# Create a coordinate map for out phase space
coord_map = CoordinateMap(proc, model, out_psl)

# call on in- and out-coordinates to build the momenta
coord_map(in_coords, out_coords)
```

- **Arguments**:
    - `in_coords`: A tuple of phase space coordinates for the incoming particles.
    - `out_coords`: A tuple of phase space coordinates for the outgoing particles.

- **Returns**:
    - A tuple where the first element is the incoming particle momenta and the second element
        is the outgoing particle momenta, both represented as tuples of four-momenta.

## Notes
- The `CoordinateMap` provides a flexible mechanism for transforming phase space coordinates
    into physically meaningful four-momenta, ensuring consistency with the scattering process,
    physics model, and phase space layout.
- The type is designed to handle both incoming and outgoing momenta, ensuring proper energy
    and momentum conservation for the process.

"""
struct CoordinateMap{P, M, PSL <: AbstractPhaseSpaceLayout} <: AbstractCoordinateMap
    proc::P
    model::M
    psl::PSL
end

# make the transform callable
@inline function (coord_map::CoordinateMap{P, M, PSL})(
        in_coords::Tuple
    ) where {P, M, PSL <: AbstractInPhaseSpaceLayout}
    return build_momenta(coord_map.proc, coord_map.model, coord_map.psl, in_coords)
end

# make the transform callable
@inline function (coord_map::CoordinateMap{P, M, PSL})(
        in_coords::Tuple, out_coords::Tuple
    ) where {P, M, PSL <: AbstractOutPhaseSpaceLayout}
    in_moms = build_momenta(
        coord_map.proc, coord_map.model, in_phase_space_layout(coord_map.psl), in_coords
    )
    return in_moms,
        build_momenta(coord_map.proc, coord_map.model, in_moms, coord_map.psl, out_coords)
end
