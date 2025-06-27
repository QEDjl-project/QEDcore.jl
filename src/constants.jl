# source: CODATA2022
const _alpha_f64 = 7.2973525643e-3
const _me_f64 = 5.1099895069e5

"""
    ALPHA

The fine-structure constant ``\\alpha \\approx 1/137`` (dimensionless), which characterizes the strength of electromagnetic interactions.

This value is defined using exactly the recommended value given in [CODATA2022](@cite), but can be safely converted to any floating-point type (e.g., `Float64(ALPHA)` or `one(Float32)*ALPHA`) in a type-stable and compile-time safe manner.
"""
ALPHA
Base.@irrational ALPHA big(_alpha_f64)

"""
    ALPHA_SQUARE

The square of the fine-structure constant, i.e. ``\\alpha^2``. Useful in second-order electromagnetic processes.

Supports type-stable conversion to all standard floating-point types.
"""
ALPHA_SQUARE
Base.@irrational ALPHA_SQUARE big((_alpha_f64)^2)

"""
    ELEMENTARY_CHARGE

The elementary charge ``e`` in natural units, defined as ``\\sqrt{4\\pi \\alpha}``.

Dimensionless in natural units; useful in calculating Coulomb interactions or electromagnetic coupling factors.

This value is defined using exactly the recommended value given in [CODATA2022](@cite), but can be safely converted to any floating-point type (e.g., `Float64(ELEMENTARY_CHARGE)` or `one(Float32)*ELEMENTARY_CHARGE`) in a type-stable and compile-time safe manner.
"""
ELEMENTARY_CHARGE
Base.@irrational ELEMENTARY_CHARGE big(sqrt(4 * pi * _alpha_f64))

"""
    ELEMENTARY_CHARGE_SQUARE

The square of the elementary charge in natural units: ``e^2``.

Commonly appears as a prefactor in scattering cross sections and Coulomb potentials.
"""
ELEMENTARY_CHARGE_SQUARE
Base.@irrational ELEMENTARY_CHARGE_SQUARE big(4 * pi * _alpha_f64)

"""
    ELECTRONMASS

Electron rest mass in **electronvolts** (eV), consistent with natural units.

This value is defined using exactly the recommended value given in [CODATA2022](@cite), but can be safely converted to any floating-point type (e.g., `Float64(ELECTRONMASS)` or `one(Float32)*ELECTRONMASS`) in a type-stable and compile-time safe manner.
"""
ELECTRONMASS
Base.@irrational ELECTRONMASS big(_me_f64) # eV

"""
    ONE_OVER_FOURPI

The value ``1 / (4\\pi)``, often used as a Coulomb prefactor in Gaussian or natural unit systems.

Convertible at compile time to all standard float types.
"""
ONE_OVER_FOURPI
Base.@irrational ONE_OVER_FOURPI big(inv(4 * pi))
