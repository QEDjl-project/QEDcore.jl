using InteractiveUtils #hide
using QEDcore # hide
# # Using Physical Constants in a Type-Stable Way

# This tutorial shows how to work with physical constants defined using `Base.@irrational`
# in a way that is **type-stable**, **efficient**, and **compile-time safe**.

# In `QEDcore.jl` the following constants are defined:

# ```julia
# ALPHA
# ALPHA_SQUARE
# ELEMENTARY_CHARGE
# ELEMENTARY_CHARGE_SQUARE
# ELECTRONMASS
# ONE_OVER_FOURPI
# ```

# These constants are defined with the exact value given by [CODATA2022](@cite), but can be used in
# any floating-point type (`Float64`, `Float32`, `Float16`) **without losing type stability**.

# ## Convert to a Different Floating-Point Type

# Use the target type constructor, like `Float32(ALPHA)`:

Float32(ALPHA)

#

Float16(ELEMENTARY_CHARGE)

# You can do this with any constant. The result is a value of the requested type, and the
# conversion happens **statically at compile time**.

# ## Automatic Conversion Using `one(T)` Pattern

# This trick is useful for writing generic functions:

T = Float32
one(T) * ALPHA

# ## Why This is Great: Compile-Time Efficiency

# These conversions are evaluated at **compile time**. That means:

# - The compiler inserts the constant value directly in machine code.
# - There’s no runtime overhead.
# - Your numerical code remains **type-stable** and **fast**.

# You can verify this using:

@code_typed Float32(ALPHA)

# You'll see the compiler inserts a literal `Float32` like `0.007297353f0`.

# ## Summary

# - Use `Float32(ALPHA)` or `one(Float32)*ALPHA` to get type-stable values.
# - All conversions happen at **compile time** — no performance hit.
# - Great for writing generic numerical code with physical constants.
