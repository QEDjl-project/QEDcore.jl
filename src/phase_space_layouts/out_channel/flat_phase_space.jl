# here we implement RAMBO

"""
    FlatPhaseSpaceLayout{INPSL} <: QEDcore.AbstractOutPhaseSpaceLayout{INPSL}

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
- `QEDcore.phase_space_dimension`: Calculates the phase space dimension based on the number of
  outgoing particles.
- `QEDcore._build_momenta`: Constructs the outgoing momenta using the provided phase space layout
  and RAMBO-based algorithms, ensuring the momenta satisfy energy and momentum conservation laws.

# Example
```julia
# Define input and process information, model, and input coordinates
psl = FlatPhaseSpaceLayout(input_psl)
build_momenta(process, model, incoming_momenta, psl, out_coords)
```

`FlatPhaseSpaceLayout` provides a robust setup for defining the final-state phase space in particle
physics simulations, allowing for modularity and compatibility with `QEDcore` routines.
"""
struct FlatPhaseSpaceLayout{INPSL} <: QEDcore.AbstractOutPhaseSpaceLayout{INPSL}
    in_psl::INPSL
end

"""
    QEDcore.phase_space_dimension(proc::AbstractProcessDefinition, model::AbstractModelDefinition, psl::FlatPhaseSpaceLayout)

Calculates the phase space dimensionality for a given process, model, and phase space layout.
This dimension is derived as four times the number of outgoing particles.
"""
function QEDcore.phase_space_dimension(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    psl::FlatPhaseSpaceLayout,
)
    return 4 * number_outgoing_particles(proc)
end

QEDcore.in_phase_space_layout(psl::FlatPhaseSpaceLayout) = psl.in_psl

function QEDcore._build_momenta(
    proc::AbstractProcessDefinition,
    model::AbstractModelDefinition,
    in_moms::Tuple,
    psl::FlatPhaseSpaceLayout,
    out_coords::Tuple,
)

    # TODO: move this to input validation
    number_outgoing_particles(proc) >= 2 || throw(
        InvalidInputError(
            "the number of particles <$(number_outgoing_particles(proc)) must be at least two",
        ),
    )
    # preparing inchannel
    Ptot = sum(in_moms)
    boost_from_rest = inv(_unsafe_rest_boost(Ptot))

    sqrt_s = sqrt(Ptot * Ptot)
    out_mass_sum = sum(mass.(outgoing_particles(proc)))

    # TODO: move this to input validation
    sqrt_s >= out_mass_sum || throw(
        InvalidInputError(
            """
            sum of the masses of the outgoing particles <$out_mass_sum> must not exceed the
            center-of-momentum energy <$sqrt_s>
            """,
        ),
    )

    out_moms = _massive_rambo_moms(out_coords, sqrt_s, mass.(outgoing_particles(proc)))

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

function _to_be_solved(xi, masses, p0s, ss)
    s = mapreduce(x -> sqrt(@inbounds x[1]^2 + xi^2 * x[2]^2), +, zip(masses, p0s))
    return s - ss
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
function _find_scaling_factor(masses, energies, ss)
    f = x -> _to_be_solved(x, masses, energies, ss)
    xi = find_zero(f, 2, Order1())
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
    a, b, c, d = single_coords
    cth = 2 * c - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * pi * d
    p0 = -log(a) - log(b)
    p1 = p0 * sth * cos(phi)
    p2 = p0 * sth * sin(phi)
    p3 = p0 * cth
    return SFourMomentum(p0, p1, p2, p3)
end

"""
    _tuple_partition_by_four(c::Tuple)

Partitions a tuple of coordinates by four, generating sub-tuples of four values each without allocating memory.

# Arguments
- `c::Tuple`: Tuple of coordinates.

# Returns
- NTuple containing four-element tuples.
"""
function _tuple_partition_by_four(c)
    N = length(c)
    m = div(N, 4)
    return NTuple{m}(c[i:(i + 3)] for i in 1:4:N)
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
    return SFourMomentum(getT(Q) / M, scale_spatial(Q, -inv(M))...)
end

function _transform2conserved(bvec, scale, mom)
    a = 1 / (1 + getT(bvec))

    spatial_mom = SVector{3}(view(mom, 2:4))
    spatial_bvec = SVector{3}(view(bvec, 2:4))

    #bq = bx * mom1 + by * mom2 + bz * mom3
    bq = LinearAlgebra.dot(spatial_bvec, spatial_mom)

    return SFourMomentum(
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
    return SFourMomentum(
        sqrt(getT(massless_mom)^2 * xi^2 + mass^2),
        xi * getX(massless_mom),
        xi * getY(massless_mom),
        xi * getZ(massless_mom),
    )
end

function _scale_rambo_moms(xi, masses, massless_moms)
    return map(x -> _scale_single_rambo_mom(xi, x...), zip(masses, massless_moms))
end

function QEDbase._phase_space_factor(
    psp::PhaseSpacePoint{PROC,MODEL,PSL}
) where {
    PROC<:AbstractProcessDefinition,MODEL<:AbstractModelDefinition,PSL<:FlatPhaseSpaceLayout
}
    # TODO: implement rambo weights here

end
