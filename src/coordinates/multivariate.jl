"""
    CoordinateSet{N, D<:Tuple}

A concrete type that represents a set of `N` coordinates, stored as a tuple `coords::D`.
This type is a subtype of `AbstractCoordinateSet{N}` and is designed to handle sets of
coordinates for use in various calculations, such as in phase space layouts.

# Fields
- `coords::D`: A tuple containing the individual coordinates. Each element of the tuple
    corresponds to a coordinate in the set.

# Constructors
- `CoordinateSet{N}(coords::D)`: Creates a `CoordinateSet` where the number of coordinates
    `N` must match the length of the tuple `coords`. If they do not match,
    an `ArgumentError` is thrown.
- `CoordinateSet(coords::D)`: Automatically infers `N` as the length of the provided tuple
    `coords`.

# Throws
- `ArgumentError` if the length of the provided tuple does not match the specified
    number `N`.
"""
struct CoordinateSet{N,D<:Tuple} <: AbstractCoordinateSet{N}
    coords::D

    function CoordinateSet{N}(coords::D) where {N,D<:Tuple}
        Nc = length(coords)
        N == Nc || throw(
            ArgumentError(
                "number of coordinates <$Nc> needs to be equal to the specified number <$N>",
            ),
        )
        return new{N,D}(coords)
    end

    # skip check if length is not given
    CoordinateSet(coords::D) where {D<:Tuple} = new{length(coords),D}(coords)
end
function CoordinateSet{N}(coords::AbstractUnivariateCoordinates...) where {N}
    return CoordinateSet{N}(coords)
end
CoordinateSet(coords::AbstractUnivariateCoordinates...) = CoordinateSet(coords)

phase_space_dimension(::CoordinateSet{N}) where {N} = N

BivariateCoordiantes(coords::Tuple) = CoordinateSet{2}(coords)
TrivariateCoordiantes(coords::Tuple) = CoordinateSet{3}(coords)
function coordinate_names(coord_set::CoordinateSet)
    return tuple(collect(coordinate_name(coord) for coord in coord_set.coords)...)
end
