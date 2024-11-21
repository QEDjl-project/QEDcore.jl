"""
    CoordinateMapCached{P,M,PSL,TM}(proc::P, model::M, psl::PSL, in_moms::TM)

A `CoordinateMapCached` represents a precomputed transformation for phase space coordinates,
where the momenta of the incoming particles are cached. This can improve performance in cases
where the incoming momenta remain constant and only the outgoing momenta need to be computed
based on new coordinates.

The cached map encapsulates the scattering process (`proc`), the physics model (`model`),
the phase space layout (`psl`), and the precomputed incoming momenta (`in_moms`).

## Fields
- `proc`: The scattering process definition, which defines the incoming and outgoing particles.
- `model`: The physics model, which governs the interaction type and momentum distributions.
- `psl`: The phase space layout, either incoming or outgoing, that maps coordinates to momenta.
- `in_moms`: A collection of precomputed four-momenta for the incoming particles, usually a `Tuple`.

## Usage

### Cached Incoming Coordinates
When a `CoordinateMapCached` build with `psl::AbstractInPhaseSpaceLayout` is called without
any arguments, it returns the precomputed incoming momenta (`in_moms`) directly. This provides
an efficient way to access the incoming particle momenta that have already been calculated and
stored in the map.

```julia
# Create a coordinate map for in phase space
in_coord_map = CoordinateMapCached(proc, model, in_psl, in_moms)

# call to return the in-momenta
in_coord_map()
```

- **Returns**: The cached tuple of four-momenta for the incoming particles.

### Outgoing Coordinates
The `CoordinateMapCached` can be called with outgoing phase space coordinates (`out_coords`).
The cached incoming momenta (`in_moms`) are used to compute the total momentum (`Ptot`), which
is then passed along with the outgoing coordinates to compute the momenta of the outgoing
particles.

```julia
# Create a coordinate map for out phase space
out_coord_map = CoordinateMapCached(proc, model, out_psl, in_moms)

# call on out coordinates to return the out-momenta
out_coord_map(out_coords)
```

- **Arguments**:
    - `out_coords`: A tuple of phase space coordinates for the outgoing particles.

- **Returns**:
    - A tuple of four-momenta for the outgoing particles, consistent with the total momentum
        derived from the cached incoming momenta.

## Notes
- **Caching**: The `CoordinateMapCached` is useful when the incoming momenta are fixed or do
    not change frequently, as it avoids recomputation by storing the incoming momenta in the
    cache.
- **Efficiency**: This caching mechanism can significantly enhance performance when repeatedly
    evaluating the scattering process with fixed incoming particles but varying outgoing
    configurations.
- The type is designed to handle both incoming and outgoing momenta, ensuring proper energy
    and momentum conservation for the process.

"""
struct CoordinateMapCached{
    P<:AbstractProcessDefinition,
    M<:AbstractModelDefinition,
    PSL<:QEDbase.AbstractPhaseSpaceLayout,
    TM,
} <: AbstractCoordinateMap
    proc::P
    model::M
    psl::PSL
    in_moms::TM
end

function CoordinateMapCached(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    psl::QEDbase.AbstractInPhaseSpaceLayout,
    in_coords::NTuple{N,T},
) where {N,T<:Real}
    in_moms = QEDbase.build_momenta(proc, model, psl, in_coords)
    return CoordinateMapCached(proc, model, psl, in_moms)
end

function CoordinateMapCached(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    psl::QEDbase.AbstractOutPhaseSpaceLayout,
    in_coords::NTuple{N,T},
) where {N,T<:Real}
    in_moms = QEDbase.build_momenta(
        proc, model, QEDbase.in_phase_space_layout(psl), in_coords
    )
    return CoordinateMapCached(proc, model, psl, in_moms)
end

# make the transform callable: for in_psl maps return the cached
@inline function (
    coord_map::CoordinateMapCached{P,M,PSL}
)() where {P,M,PSL<:QEDbase.AbstractInPhaseSpaceLayout}
    return getfield(coord_map, :in_moms)
end

# make the transform callable: for out_psl maps
@inline function (coord_map::CoordinateMapCached{P,M,PSL})(
    out_coords::Tuple
) where {P,M,PSL<:QEDbase.AbstractOutPhaseSpaceLayout}
    in_moms = coord_map.in_moms
    return QEDbase.build_momenta(
        coord_map.proc, coord_map.model, in_moms, coord_map.psl, out_coords
    )
end
