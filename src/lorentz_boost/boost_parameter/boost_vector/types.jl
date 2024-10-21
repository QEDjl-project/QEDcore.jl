"""
    AbstractBoostVector <: QEDbase.AbstractBoostParameter

An abstract base type representing vector-like boost parameters, used to model Lorentz boosts
in any spatial dimension.

`AbstractBoostVector` extends `QEDbase.AbstractBoostParameter` and provides the framework for
describing boosts that act in multiple spatial dimensions simultaneously, typically in
three-dimensional space. This type is designed to support vector representations of
velocities (in units of the speed of light) associated with Lorentz transformations in
special relativity.

## Usage

Concrete subtypes of `AbstractBoostVector` represent specific boost vectors that describe
the velocity components in each spatial dimension, such as [`BetaVector`](@ref). These boost
vectors are commonly used in transformations of four-vectors (e.g., four-momentum,
four-position) between different reference frames.

For example:
- [`BetaVector{T}`](@ref): A concrete subtype representing a boost vector with velocity components ``\\beta_x``, ``\\beta_y``, and ``\\beta_z`` (in units of the speed of light).

"""
abstract type AbstractBoostVector <: QEDbase.AbstractBoostParameter end
