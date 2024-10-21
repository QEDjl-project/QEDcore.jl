
abstract type AbstractCoordinateSet{N} end

"""
    coordinate_names(::AbstractCoordinateSet)

"""
function coordinate_names end

# TODO: think about moving to singular "AbstractCoordinate"
const AbstractUnivariateCoordinates = AbstractCoordinateSet{1}

"""
    coordinate_name(::AbstractUnivariateCoordinates)

"""
function coordinate_name end
@inline coordinate_names(coord::AbstractUnivariateCoordinates) = (coordinate_name(coord),)
