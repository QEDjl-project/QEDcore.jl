"""
    _bisection(f::Function, a::T, b::T; tol::T = eps(T), max_iter::Int = 100) where {T}

Compute the root of a function `f: T -> T` between the bounds `a` and `b` with tolerance `tol` (i.e., `abs(f(root)) <= tol || (b-a)/2==0`).
This assumes that there is exactly one root and aborts if this root has not been found after `max_iter` iterations, which is
100 by default.
"""
function _bisection(f::Function, a::T, b::T; tol::T = eps(T), max_iter::Int = 100) where {T}
    fa = f(a)
    fb = f(b)

    if fa * fb > 0
        throw(InvalidInputError("function must have opposite signs at the interval endpoints"))
    end

    for _ in 1:max_iter
        center = (a + b) / 2
        fc = f(center)

        if abs(fc) <= tol || center == a || center == b
            return center
        end

        if fa * fc < 0
            b = center
            fb = fc
        else
            a = center
            fa = fc
        end
    end

    error("bisection did not converge")
end

"""
    sq_diff(x, y)

Calculate `x^2 - y^2` with more precision than default when x ≈ y.
"""
function sq_diff(x::T1, y::T2) where {T1, T2}
    T = promote_type(T1, T2)
    return (T(x) + T(y)) * (T(x) - T(y))
end

"""
    sq_diff_sqrt(x, y)

Calculate `sqrt(x^2 - y^2)` with more precision than default when x ≈ y.
"""
function sq_diff_sqrt(x::T1, y::T2) where {T1, T2}
    T = promote_type(T1, T2)
    return sqrt(T(x) + T(y)) * sqrt(T(x) - T(y))
end
