
"""
    AbstractCoordinateSet{N}

An abstract type representing a set of coordinates with `N` elements.

Subtypes of `AbstractCoordinateSet` used to define specific coordinate systems or layouts
used to describe physical processes, such as phase-space parametrizations for scattering events.
The parameter `N` specifies the number of coordinates in the set, where `N` can be one for
univariate coordinates (e.g., energy, rapidity) or higher for more complex systems.

# Type Parameters
- `N`: The number of coordinates in the set.

# Intended Usage
Subtypes of `AbstractCoordinateSet{N}` implement the structure and behavior for specific coordinate
systems. Functions like `coordinate_names` and `coordinate_name` are used to retrieve human-readable
names for the coordinates in the set.
"""
abstract type AbstractCoordinateSet{N} end

"""
    coordinate_names(coord_set::AbstractCoordinateSet)

Retrieve the names of all coordinates in a given coordinate set.

# Arguments
- `coord_set::AbstractCoordinateSet`: A coordinate set, which is a container for multiple
coordinates.

# Returns
- A tuple of strings, where each string is the name of a coordinate in the set, including its particle index if available.

This function is typically implemented for subtypes of `AbstractCoordinateSet` and is used to return human-readable names for each coordinate in the set.
"""
function coordinate_names end

# TODO: think about moving to singular "AbstractCoordinate"
const AbstractUnivariateCoordinates = AbstractCoordinateSet{1}

"""
    coordinate_name(coord::AbstractUnivariateCoordinates)

Retrieve the name of a single univariate coordinate.

# Arguments
- `coord::AbstractUnivariateCoordinates`: A single univariate coordinate, which is a subtype of `AbstractCoordinateSet{1}` representing a coordinate system with one degree of freedom.

# Returns
- A string representing the name of the coordinate.

This function provides a human-readable label for a single coordinate and is generally used for naming individual coordinates in a scattering process or phase space.
"""
function coordinate_name end
@inline coordinate_names(coord::AbstractUnivariateCoordinates) = (coordinate_name(coord),)
