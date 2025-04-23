# here we implement RAMBO

"""
    FlatPhaseSpaceLayout{INPSL} <: QEDbase.AbstractOutPhaseSpaceLayout{INPSL}

Defines a flat phase space layout for generating outgoing particle momenta. This layout assumes
that outgoing particles are uniformly distributed in phase space, subject to constraints on
momentum and energy conservation. This is accomplished by exploiting the RAMBO [Kleiss:1985gy](@cite) (Random Momenta Beautifully Organized)
algorithm to generate the momenta, supporting both massless
and massive particles.

# Fields
- `in_psl::INPSL`: Input phase space layout, which provides the initial state layout for the
  process. This field links to the input configuration, making it accessible for consistency
  across related calculations.

# Usage
`FlatPhaseSpaceLayout` is commonly used in high-energy physics to define phase space configurations
for event generators and cross-section calculations. Key functions include:
- `QEDbase.phase_space_dimension`: Calculates the phase space dimension based on the number of
  outgoing particles.
- `QEDbase._build_momenta`: Constructs the outgoing momenta using the provided phase space layout
  and RAMBO-based algorithms, ensuring the momenta satisfy energy and momentum conservation laws.

# Example
```julia
# Define input and process information, model, and input coordinates
psl = FlatPhaseSpaceLayout(input_psl)
build_momenta(process, model, incoming_momenta, psl, out_coords)
```

`FlatPhaseSpaceLayout` provides a robust setup for defining the final-state phase space in particle
physics simulations, allowing for modularity and compatibility with `QEDbase` routines.
"""
struct FlatPhaseSpaceLayout{INPSL} <: QEDbase.AbstractOutPhaseSpaceLayout{INPSL}
    in_psl::INPSL
end

Base.broadcastable(psl::FlatPhaseSpaceLayout) = Ref(psl)

function QEDbase.phase_space_dimension(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::FlatPhaseSpaceLayout,
    )
    return 4 * number_outgoing_particles(proc)
end

QEDbase.in_phase_space_layout(psl::FlatPhaseSpaceLayout) = psl.in_psl

function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        in_moms::Tuple,
        psl::FlatPhaseSpaceLayout,
        out_coords::Tuple,
    )
    T = eltype(eltype(in_moms))

    # TODO: move this to input validation
    number_outgoing_particles(proc) >= 2 || throw(
        InvalidInputError(
            "the number of particles must be at least two",
        ),
    )
    # preparing inchannel
    Ptot = sum(in_moms)
    boost_from_rest = inv(_unsafe_rest_boost(Ptot))

    sqrt_s = sqrt(Ptot * Ptot)
    out_mass_sum = sum(mass.(T, outgoing_particles(proc)))

    # TODO: move this to input validation
    sqrt_s >= out_mass_sum || throw(
        InvalidInputError(
            """
            sum of the masses of the outgoing particles must not exceed the
            center-of-momentum energy
            """,
        ),
    )

    out_moms = _massive_rambo_moms(out_coords, sqrt_s, mass.(T, outgoing_particles(proc)))

    return boost_from_rest.(out_moms)
end

"""

   _massive_rambo_moms(c::Tuple, ss::Float64, masses::Tuple)

Computes the massive outgoing momenta using RAMBO for given coordinates, center-of-momentum energy, and particle masses.

# Arguments
- `c::Tuple`: Tuple of uniformly distributed coordinates.
- `ss::Float64`: Center-of-momentum energy.
- `masses::Tuple` Vector of particle masses for each outgoing particle.

# Returns
- A tuple of four-momenta for outgoing particles.
"""
function _massive_rambo_moms(c, ss, masses)
    massless_moms = _massless_rambo_moms(c, ss)
    energies = getT.(massless_moms)
    xi = _find_scaling_factor(masses, energies, ss)
    return _scale_rambo_moms(xi, masses, massless_moms)
end

@inline function _to_be_solved(xi::T1, masses::NTuple{N, T2}, p0s::NTuple{N, T3}, ss::T4)::T1 where {N, T1, T2, T3, T4}
    T = promote_type(T1, T2, T3, T4)
    s = sum(hypot.(masses, xi .* p0s))
    return T(s - ss)
end

"""
    _bisection(f::Function, a::T, b::T; tol::T = eps(T), max_iter::Int = 100) where {T}

Compute the root of a function `f: T -> T` between the bounds `a` and `b` with tolerance `tol` (i.e., `abs(f(root)) <= tol`).
This assumes that there is exactly one root and aborts if this root has not been found after `max_iter` iterations, which is
100 by default.
"""
function _bisection(f::Function, a::T, b::T; tol::T = eps(T), max_iter::Int = 100) where {T}
    fa = f(a)
    fb = f(b)

    if fa * fb > 0
        error("function must have opposite signs at the interval endpoints")
    end

    for _ in 1:max_iter
        center = (a + b) / 2
        fc = f(center)

        if abs(fc) < tol || (b - a) / 2 < tol
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

    error("bisection method did not converge")
end


"""
    _find_scaling_factor(masses::Tuple, energies::Tuple, ss::Float64)

Finds a scaling factor for particle momenta to enforce conservation of energy-momentum in massive RAMBO.

# Arguments
- `masses::Tuple`: Vector of outgoing particle masses.
- `energies::Tuple`: Vector of outgoing particle energies.
- `ss::Float64`: Center-of-momentum energy.

# Returns
- The computed scaling factor as a float.
"""
function _find_scaling_factor(masses::NTuple{N, T1}, energies::NTuple{N, T2}, ss::T3) where {N, T1, T2, T3}
    T = promote_type(T1, T2, T3)
    f = x -> _to_be_solved(x, masses, energies, ss)
    xi = _bisection(f, T(0), T(5))
    return xi
end

"""
    _single_rambo_mom(single_coords::Tuple)

Generates the four-momentum of a single particle from uniformly distributed coordinates.
Assumes massless particle as an intermediate step in RAMBO.

# Arguments
- `single_coords::Tuple`: Tuple of coordinates used to generate the particle's momentum.

# Returns
- A four-momentum (`SFourMomentum`) for a single particle.
"""
function _single_rambo_mom(single_coords)
    T = eltype(single_coords)
    a, b, c, d = single_coords
    cth = 2 * c - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * T(pi) * d
    p0 = -log(a) - log(b)
    p1 = p0 * sth * cos(phi)
    p2 = p0 * sth * sin(phi)
    p3 = p0 * cth
    return SFourMomentum{T}(p0, p1, p2, p3)
end

"""
    _tuple_partition_by_four(c::Tuple)

Partitions a tuple of coordinates by four, generating sub-tuples of four values each without allocating memory and in a type stable way.
Currently overloaded for tuples of lengths 4..40.

# Arguments
- `c::Tuple`: Tuple of coordinates.

# Returns
- NTuple containing four-element tuples.
"""
function _tuple_partition_by_four end

# block to generate a tuple partitioning function for reasonably sized tuples
# this is necessary for type stability on GPU
for O in 1:10
    I = 4 * O

    constructor_string = "tuple("
    for i in 1:4:I
        constructor_string *= "(c[$i], c[$(i + 1)], c[$(i + 2)], c[$(i + 3)]),"
    end
    constructor_string *= ")"
    constructor = Meta.parse(constructor_string)

    @eval function _tuple_partition_by_four(c::NTuple{$I, T})::NTuple{$O, NTuple{4, T}} where {T}
        return $constructor
    end
end

@inline function scale_spatial(lv, fac)
    return fac * SVector{3}(view(lv, 2:4))
end

"""
    _unconserved_momenta(c::Tuple)

Builds an initial set of momenta from uniform coordinates that do not necessarily conserve energy-momentum.

# Arguments
- `c::Tuple`: Tuple of uniformly distributed coordinates.

# Returns
- Tuple of four-momenta representing unconserved momenta.
"""
function _unconserved_momenta(c)
    return map(_single_rambo_mom, _tuple_partition_by_four(c))
end

function _rambo_bvec(Q, M)
    T = promote_type(eltype(Q), eltype(M))
    return SFourMomentum{T}(getT(Q) / M, scale_spatial(Q, -inv(M))...)
end

function _transform2conserved(bvec, scale, mom)
    T = promote_type(eltype(bvec), typeof(scale), eltype(mom))

    a = 1 / (1 + getT(bvec))

    spatial_mom = SVector{3}(view(mom, 2:4))
    spatial_bvec = SVector{3}(view(bvec, 2:4))

    #bq = bx * mom1 + by * mom2 + bz * mom3
    bq = LinearAlgebra.dot(spatial_bvec, spatial_mom)

    return SFourMomentum{T}(
        scale * (getT(bvec) * getT(mom) + bq),
        (scale * (spatial_mom + (getT(mom) + a * bq) * spatial_bvec))...,
    )
end

"""
    _massless_rambo_moms(c::Tuple, ss::Float64)

Generates a set of massless momenta based on uniformly distributed coordinates.
Ensures energy-momentum conservation.

# Arguments
- `c::Tuple`: Tuple of uniformly distributed coordinates.
- `ss::Float64`: Center-of-momentum energy.

# Returns
- Tuple of massless four-momenta, which satisfy energy-momentum conservation.
"""
function _massless_rambo_moms(c, ss)
    _moms = _unconserved_momenta(c)
    Q = sum(_moms)
    M = getMass(Q)
    b = _rambo_bvec(Q, M)
    scale = ss / M

    return map(x -> _transform2conserved(b, scale, x), _moms)
end

function _scale_single_rambo_mom(xi, mass, massless_mom)
    T = typeof(xi)
    return SFourMomentum{T}(
        hypot(getT(massless_mom) * xi, mass),
        xi * getX(massless_mom),
        xi * getY(massless_mom),
        xi * getZ(massless_mom),
    )
end

function _scale_rambo_moms(xi, masses, massless_moms)
    n = size(masses, 1)
    return ntuple(i -> _scale_single_rambo_mom(xi, masses[i], massless_moms[i]), n)
    #return Tuple(map(x -> _scale_single_rambo_mom(xi, x...), zip(masses, massless_moms)))
end

# Kleiss 1985: 2.14
function _massless_rambo_weight(ss, n)
    return (pi / 2)^(n - 1) * ss^(2 * n - 4) / (factorial(n - 1) * factorial(n - 2))
end

# Kleiss 1985: 4.11
function _massive_rambo_weight(ss, out_moms, n)
    Es = getE.(out_moms)
    rhos = getRho.(out_moms)

    rhos_over_Es = rhos ./ Es

    fac1 = prod(rhos_over_Es)
    fac2 = inv(sum(rhos_over_Es .* rhos))
    fac3 = sum(rhos)^(2 * n - 3)
    fac4 = ss^(3 - 2 * n)

    return fac1 * fac2 * fac3 * fac4
end

function _center_of_momentum_energy(psp)
    P_total = sum(momenta(psp, Incoming()))
    return getMass(P_total)
end

function QEDbase._phase_space_factor(
        psp::PhaseSpacePoint{PROC, MODEL, PSL}
    ) where {
        PROC <: AbstractProcessDefinition, MODEL <: AbstractModelDefinition, PSL <: FlatPhaseSpaceLayout,
    }
    ss = _center_of_momentum_energy(psp)
    n = number_incoming_particles(psp)
    out_moms = momenta(psp, Outgoing())

    massless_weight = _massless_rambo_weight(ss, n)
    massive_weight = _massive_rambo_weight(ss, out_moms, n)

    return massless_weight * massive_weight
end
