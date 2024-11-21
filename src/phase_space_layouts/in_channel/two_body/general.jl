
abstract type AbstractPerturbativeModel <: AbstractModelDefinition end

"""
    AbstractTwoBodyInPhaseSpaceLayout <: AbstractInPhaseSpaceLayout

An abstract type representing an incoming phase space layout specifically designed for two-body
systems in high-energy physics. This type is a specialized subtype of `AbstractInPhaseSpaceLayout`,
focusing on scenarios where two particles are incoming and participate in a scattering or
decay process.

Concrete subtypes of `AbstractTwoBodyInPhaseSpaceLayout` define the parameterization of the
momenta of the two incoming particles, such as by [`Energy`](@ref), [`Rapidity`](@ref), or other kinematic coordinates.
These layouts allow for consistent calculation of the system's phase-space properties under
conservation laws, facilitating method dispatch in functions like [`QEDbase._build_momenta`](@extref).

This type serves as a foundation for implementing concrete layouts, ensuring that two-body
processes can be modeled flexibly and accurately within phase space calculations.

# See Also
- [`QEDbase.AbstractInPhaseSpaceLayout`](@extref): A broader type for general incoming phase space layouts.
- [`AbstractTwoBodyRestSystem`](@ref): A subtype representing two-body systems where one
    particle is at rest.
"""
abstract type AbstractTwoBodyInPhaseSpaceLayout <: QEDbase.AbstractInPhaseSpaceLayout end
