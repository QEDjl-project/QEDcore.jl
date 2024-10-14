# # Phase Space Definition

# !!! note
#     [`PhasespaceDefinition`](@ref)s are to be reworked (see this [issue](https://github.com/QEDjl-project/QEDcore.jl/issues/50)). Therefore, this manual is very rudimentary for the moment.
#

# A [`PhasespaceDefinition`](@ref) is a representation of a phase space's layout. It is a singleton type definition and has an [`AbstractCoordinateSystem`](@extref QEDbase.AbstractCoordinateSystem) and an [`AbstractFrameOfReference`](@extref QEDbase.AbstractFrameOfReference).

using QEDcore
ps_def = PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame())

# The phase space definition is used in [`PhaseSpacePoint`](@ref)s for dispatching in some of the cross-section interface functions.
