module QEDcore

# lorentz vectors
export SLorentzVector, MLorentzVector

# four momenta
export SFourMomentum, MFourMomentum

# four momenta
export Boost
export BetaX, BetaY, BetaZ, BetaVector

# spinors
export BiSpinor, AdjointBiSpinor, DiracMatrix

# gamma matrices
export gamma, GAMMA, DiracGammaRepresentation, slashed

# particle types
export FermionLike, Fermion, AntiFermion, MajoranaFermion
export BosonLike, Boson, AntiBoson, MajoranaBoson
export Electron, Positron, Photon

# particle base states
export base_state

# phase space layout
export CoordinateMap
export CoordinateMapCached

# phase space
export SphericalCoordinateSystem
export CenterOfMomentumFrame, ElectronRestFrame
export PhasespaceDefinition
export ParticleStateful, PhaseSpacePoint, InPhaseSpacePoint, OutPhaseSpacePoint
export spin, polarization, momenta, getindex

using Reexport
using DocStringExtensions
using StaticArrays
using SimpleTraits

@reexport using QEDbase

include("algebraic_objects/dirac_tensors/types.jl")
include("algebraic_objects/dirac_tensors/multiplication.jl")

include("coordinate_map/map.jl")
include("coordinate_map/cached.jl")

include("phase_spaces/types.jl")
include("phase_spaces/access.jl")
include("phase_spaces/create.jl")
include("phase_spaces/print.jl")
include("phase_spaces/utility.jl")

include("algebraic_objects/four_momentum.jl")
include("algebraic_objects/lorentz_vector.jl")
include("algebraic_objects/gamma_matrices.jl")

include("lorentz_boost/types.jl")
include("lorentz_boost/boost_parameter/boost_axis/types.jl")
include("lorentz_boost/boost_parameter/boost_axis/convert.jl")
include("lorentz_boost/boost_parameter/boost_axis/beta.jl")
include("lorentz_boost/boost_parameter/boost_vector/types.jl")
include("lorentz_boost/boost_parameter/boost_vector/beta.jl")
include("lorentz_boost/boost_parameter/boost_vector/utils.jl")

include("particles/particle_types.jl")
include("particles/propagators.jl")
include("particles/states.jl")

end
