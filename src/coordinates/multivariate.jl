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
