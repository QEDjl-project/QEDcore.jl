# # Using Physical Constants in a Type-Stable Way

# This tutorial demonstrates how to work with physical constants defined using `Base.@irrational`
# in a **type-preserving** way, i.e., ensuring that calculations don't accidentally upcast to a
# different floating-point type due to the constant's default type.

# In `QEDcore.jl`, the following constants are defined:

# ```julia
# ALPHA
# ALPHA_SQUARE
# ELEMENTARY_CHARGE
# ELEMENTARY_CHARGE_SQUARE
# ELECTRONMASS
# ONE_OVER_FOURPI
# ```

# These constants are defined with the exact values given by [CODATA2022](@cite), and they can be
# used with any floating-point type (`Float64`, `Float32`, `Float16`) **without requiring runtime conversion**.

using InteractiveUtils #hide
using QEDcore #hide

# ## Converting to a Different Floating-Point Type

# You can explicitly convert a constant to a target type using a constructor, like:

Float32(ALPHA)

# or:

Float16(ELEMENTARY_CHARGE)

# This works with any of the constants. The result is a value of the requested type,
# and the conversion occurs **at compile time**:

@code_typed Float32(ALPHA)

# This adds no runtime overhead.

# ## Automatic Conversion during Arithmetic Operations

# For `Base.AbstractIrrational` types, arithmetic operations are defined such that
# the result adopts the type of the other operand. For example:

typeof(1.0 * ALPHA)

#

typeof(1.0f0 * ALPHA)
