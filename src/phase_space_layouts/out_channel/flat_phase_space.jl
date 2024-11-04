# here we implement RAMBO

struct FlatPhaseSpaceLayout{INPSL} <: QEDcore.AbstractOutPhaseSpaceLayout{INPSL}
    in_psl::INPSL
end

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
    # preparing inchannel
    # TODO: build trafo to boost the result back to the Ptot system
    Ptot = sum(in_moms)
    sqrt_s = Base.sqrt_llvm(Ptot * Ptot)

    return out_moms = _massive_rambo_moms(
        out_coords, sqrt_s, mass.(outgoing_particles(proc))
    )

    # TODO: trafo out_moms to the Ptot system
end

function _massive_rambo_moms(c, ss, masses)
    massless_moms = _massless_rambo_moms(c, ss)
    energies = getT.(massless_moms)
    xi = _find_scaling_factor(masses, energies, ss)
    return _scale_rambo_moms(xi, masses, massless_moms)
end

function _to_be_solved(xi, masses, p0s, ss)
    s = mapreduce(
        x -> Base.sqrt_llvm(@inbounds x[1]^2 + xi^2 * x[2]^2), +, zip(masses, p0s)
    )
    return s - ss
end

function _find_scaling_factor(masses, energies, ss)
    f = x -> _to_be_solved(x, masses, energies, ss)
    xi = find_zero(f, 2, Order1())
    return xi
end

"""

massless momentum of a single particle from uniformly distributed coordinates
"""
function _single_rambo_mom(single_coords)
    a, b, c, d = single_coords
    cth = 2 * c - 1
    sth = Base.sqrt_llvm(1 - cth^2)
    phi = 2 * pi * d
    p0 = -log(a) - log(b)
    p1 = p0 * sth * cos(phi)
    p2 = p0 * sth * sin(phi)
    p3 = p0 * cth
    return SFourMomentum(p0, p1, p2, p3)
end

"""

build tuple of partition by four of a given tuple
"""
function _tuple_partition_by_four(c)
    N = length(c)
    m = Int(N / 4)
    return NTuple{m}(c[i:(i + 4)] for i in 1:m)
end

@inline function scale_spatial(lv, fac)
    return fac * SVector{3}(view(lv, 2:4))
end

"""
build momenta from uniform coordinates, which not necessarily satisfies energy-momentum conservation
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

function _massless_rambo_moms(c, ss)
    # use moms_test4 here
    _moms = _unconserved_momenta(c)
    Q = sum(_moms)
    M = getMass(Q)
    b = _rambo_bvec(Q, M)
    scale = ss / M

    return map(x -> _transform2conserved(b, scale, x), _moms)
end

function _scale_single_rambo_mom(xi, mass, massless_mom)
    return SFourMomentum(
        Base.sqrt_llvm(getT(massless_mom)^2 * xi^2 + mass^2),
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
