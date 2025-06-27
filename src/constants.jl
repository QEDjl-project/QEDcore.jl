# source: CODATA 2022
const _alpha_f64 = 7.2973525643e-3
const _me_f64 = 5.1099895069e5

Base.@irrational ALPHA big(_alpha_f64)
Base.@irrational ALPHA_SQUARE big((_alpha_f64)^2)
Base.@irrational ELEMENTARY_CHARGE big(sqrt(4 * pi * _alpha_f64))
Base.@irrational ELEMENTARY_CHARGE_SQUARE big(4 * pi * _alpha_f64)
Base.@irrational ELECTRONMASS big(_me_f64) # eV
Base.@irrational ONE_OVER_FOURPI big(inv(4 * pi))
