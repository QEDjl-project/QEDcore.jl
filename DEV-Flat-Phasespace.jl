### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 219d5c8e-96c4-11ef-350b-691d878780a3
begin
    using Pkg: Pkg
    Pkg.activate(".")

    using QEDcore

    Pkg.add("QEDprocesses")
    using QEDprocesses

    using Plots
    using BenchmarkTools
    Pkg.add("ForwardDiff")
    using ForwardDiff

    Pkg.add("Roots")
    using Roots

    using Random

    Pkg.add("RandomExtensions")
    using RandomExtensions

    using StaticArrays
    using LinearAlgebra
end

# ╔═╡ 3fd1f0e2-7733-4306-8fd6-6d5edb5afd1b
in_psl = TwoBodyRestSystem(1, Energy(2))

# ╔═╡ 863dc644-8731-45fe-aa71-1ca3adbc0e3d
PROC = ScatteringProcess(
    (Electron(), Positron()),
    (
        Electron(),
        Positron(),
        Electron(),
        Positron(),
        Electron(),
        Positron(),
        Electron(),
        Positron(),
        Electron(),
        Positron(),
    ),
)

# ╔═╡ c3086d07-b18f-41c8-99b6-06ec6b909b95
struct pQED <: QEDcore.AbstractPerturbativeModel end

# ╔═╡ 85c4dd62-1b6a-40e6-b793-626538fe6e39
MODEL = pQED()

# ╔═╡ 0dbbeec8-8bdd-4ec8-b8b9-4c006fe552b9
coord_map = CoordinateMap(PROC, MODEL, in_psl)

# ╔═╡ eb050099-0431-44ce-b6b1-5645fbd3d17a
coord_map((10,))

# ╔═╡ 2c58278c-955c-4232-a7c9-7585794987e6
md"# New implementation"

# ╔═╡ c9acc293-a602-483b-828f-0f3db765d1a5
begin
    C = rand(NTuple{4 * 8})
    ct = rand(NTuple{4})
end

# ╔═╡ f39a8e46-afc9-41ba-9a25-1987facd8a65
a1, a2, a3, a4 = ct

# ╔═╡ 681e790f-892a-44c1-b4a6-4001a6685c5d
function mom_test(a, b, c, d)
    cth = 2 * c - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * pi * d
    sphi, cphi = sincos(phi)
    p0 = -log(a) - log(b)
    p1 = p0 * sth * cphi
    p2 = p0 * sth * sphi
    p3 = p0 * cth
    return SFourMomentum(p0, p1, p2, p3)
end

# ╔═╡ b01178b9-48bd-4cac-976f-2edd9becc1f9
function mom_test(abcd)
    a, b, c, d = abcd
    cth = 2 * c - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * pi * d
    p0 = -log(a) - log(b)
    p1 = p0 * sth * cos(phi)
    p2 = p0 * sth * sin(phi)
    p3 = p0 * cth
    return SFourMomentum(p0, p1, p2, p3)
end

# ╔═╡ 9a834479-9c4c-41aa-9850-48e9015b10e3
@code_lowered mom_test(ct)

# ╔═╡ 10d696f6-4ffd-4ede-a796-b39943bb9815
@benchmark mom_test($a1, $a2, $a3, $a4)

# ╔═╡ b5d7340e-a2f4-4b9c-8253-29505fc97c04
@benchmark mom_test($ct)

# ╔═╡ 273f91a6-9ca4-4acc-a561-5be418b7d328
begin
    function moms_test4(co0o0)
        return map(mom_test, tuple_partition_by_four(co0o0))
    end

    function tuple_partition_by_four(cooo)
        N = length(cooo)
        m = Int(N / 4)
        return NTuple{m}(cooo[i:(i + 4)] for i in 1:m)
    end
end

# ╔═╡ e02c75f1-216b-4c15-8ec6-04175945f86e
@benchmark moms_test4($C)

# ╔═╡ 5d04cdd3-ea72-46ad-b3e7-25457f3590f0
C_vec = tuple_partition_by_four(C)

# ╔═╡ 7cb6054a-767b-4c4e-90f0-8ba405d0198b
@code_warntype tuple_partition_by_four(C)

# ╔═╡ 918842aa-768b-47e5-a13a-44a9a413f1c3
@benchmark tuple_partition_by_four($C)

# ╔═╡ fef93b7a-baea-487d-98a1-263990755fac
mom = rand(SFourMomentum)

# ╔═╡ 41e8aec4-9915-4c72-baf5-84bfe7f6af21

# ╔═╡ 8af059bd-cc9c-4fd8-bf95-76ab2b3035ec
f = 4

# ╔═╡ d000188f-672a-4ea2-a0dc-dd06b07000b7
mom1, mom2 = rand(SFourMomentum, 2)

# ╔═╡ 7766e428-0fe1-4ab3-bd76-efc359e0a0ff
begin
    vec1 = SVector{3}(view(mom1, 2:4))
    vec2 = SVector{3}(view(mom2, 2:4))
end

# ╔═╡ 0dcc7bb0-6678-47e1-b84c-200168c04286
LinearAlgebra.dot(vec1, vec2)

# ╔═╡ e1cc5e2f-e247-482e-8e50-ac9710d861c4
md"# Out phase-space layout"

# ╔═╡ 2ed540e5-8d81-4f02-af4c-03b69d48036d
begin
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
end

# ╔═╡ 2879f73e-9730-4d20-9035-b8c38001ccb8
begin
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
        #sum = 0.0
        #for (i, E) in enumerate(p0s)
        #    sum += sqrt(masses[i]^2 + xi^2 * E^2)
        #end
        return s - ss
    end

    function _find_scaling_factor(masses, energies, ss)
        f = x -> _to_be_solved(x, masses, energies, ss)
        #xi = find_zero((f, first_derivative(f)), ss, Roots.Newton())
        #xi = find_zero(f, (1e-12, 2), Roots.Bisection())
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
        PROC<:AbstractProcessDefinition,
        MODEL<:AbstractModelDefinition,
        PSL<:FlatPhaseSpaceLayout,
    }
        # TODO: implement rambo weights here

    end
end

# ╔═╡ af3e6fb3-132a-4064-b791-4d39fa22981b
@benchmark scale_spatial($mom, $f)

# ╔═╡ 9eb6d679-3832-4297-a871-f2257a37602c
begin
    @show PROC
    @show MODEL
    @show in_psl
    out_psl = FlatPhaseSpaceLayout(in_psl)

    Ein = 10.0
    in_moms = (
        SFourMomentum(Ein, 0, 0, sqrt(10^2 - 1)), SFourMomentum(Ein, 0, 0, -sqrt(10^2 - 1))
    )

    out_coords = Tuple(rand(4 * 10))
    out_moms = build_momenta(PROC, MODEL, in_moms, out_psl, out_coords)
end

# ╔═╡ 3e35f741-12be-45d8-9a08-0a663824f0d5
begin
    out_coord_map = CoordinateMap(PROC, MODEL, out_psl)

    #out_coord_map((10,),out_coords)

    build_momenta(PROC, MODEL, in_moms, out_psl, out_coords)
end

# ╔═╡ a1767185-78e0-47a4-bb17-59155b26b7f7
begin
    out_en = getT.(out_moms)
end

# ╔═╡ 79de0ef8-1658-4e2c-827d-3ec1f2f30378
@benchmark build_momenta($PROC, $MODEL, $in_moms, $out_psl, $out_coords)

# ╔═╡ ca27253f-18c4-442d-bf29-e85ba3dadf3c
begin
    MASSES = mass.(outgoing_particles(PROC))
    PTOT = sum(in_moms)
    SS = sqrt(PTOT * PTOT)

    @code_typed _massive_rambo_moms(out_coords, SS, MASSES)
end

# ╔═╡ c58ff9b5-6655-4994-8556-ee69ed3856b7
@benchmark _find_scaling_factor($MASSES, $out_en, $SS)

# ╔═╡ 0fe68746-2ed3-42c6-babc-c231affe301f
@benchmark _massless_rambo_moms($out_coords, $SS)

# ╔═╡ 602c3751-9c8d-460a-b069-dfa5989a5cf5
@code_warntype _massless_rambo_moms(out_coords, SS)

# ╔═╡ 69571f69-fdae-4de2-9a07-f2550e01a8ea
@code_typed QEDcore._build_momenta(PROC, MODEL, in_moms, out_psl, out_coords)

# ╔═╡ cee050e0-f14d-4252-866a-8d399588f0a7
getMass.(in_moms)

# ╔═╡ ee4abdbc-bb3e-4414-b0e5-872c7ff5ee6f
getMass.(out_moms)

# ╔═╡ 6e35d0c1-3bfa-415c-b747-2b6604e9f12a
sum(in_moms) - sum(out_moms)

# ╔═╡ bb031fdc-35c7-4b97-96aa-aa0d05a34008
in_moms

# ╔═╡ ffa023d5-1798-4b85-9dd2-1e878f87c664
out_moms

# ╔═╡ f6450a10-4227-4a87-8703-e62014f22137
N = 6

# ╔═╡ 0d199db9-9a9c-435f-b4aa-c4eaf8955cb7
@benchmark rand($SFourMomentum, $N)

# ╔═╡ Cell order:
# ╠═219d5c8e-96c4-11ef-350b-691d878780a3
# ╠═3fd1f0e2-7733-4306-8fd6-6d5edb5afd1b
# ╠═863dc644-8731-45fe-aa71-1ca3adbc0e3d
# ╠═c3086d07-b18f-41c8-99b6-06ec6b909b95
# ╠═85c4dd62-1b6a-40e6-b793-626538fe6e39
# ╠═0dbbeec8-8bdd-4ec8-b8b9-4c006fe552b9
# ╠═eb050099-0431-44ce-b6b1-5645fbd3d17a
# ╠═2c58278c-955c-4232-a7c9-7585794987e6
# ╠═c9acc293-a602-483b-828f-0f3db765d1a5
# ╠═9a834479-9c4c-41aa-9850-48e9015b10e3
# ╠═f39a8e46-afc9-41ba-9a25-1987facd8a65
# ╠═681e790f-892a-44c1-b4a6-4001a6685c5d
# ╠═b01178b9-48bd-4cac-976f-2edd9becc1f9
# ╠═10d696f6-4ffd-4ede-a796-b39943bb9815
# ╠═b5d7340e-a2f4-4b9c-8253-29505fc97c04
# ╠═273f91a6-9ca4-4acc-a561-5be418b7d328
# ╠═e02c75f1-216b-4c15-8ec6-04175945f86e
# ╠═5d04cdd3-ea72-46ad-b3e7-25457f3590f0
# ╠═7cb6054a-767b-4c4e-90f0-8ba405d0198b
# ╠═918842aa-768b-47e5-a13a-44a9a413f1c3
# ╠═fef93b7a-baea-487d-98a1-263990755fac
# ╠═41e8aec4-9915-4c72-baf5-84bfe7f6af21
# ╠═8af059bd-cc9c-4fd8-bf95-76ab2b3035ec
# ╠═af3e6fb3-132a-4064-b791-4d39fa22981b
# ╠═d000188f-672a-4ea2-a0dc-dd06b07000b7
# ╠═7766e428-0fe1-4ab3-bd76-efc359e0a0ff
# ╠═0dcc7bb0-6678-47e1-b84c-200168c04286
# ╟─e1cc5e2f-e247-482e-8e50-ac9710d861c4
# ╠═2ed540e5-8d81-4f02-af4c-03b69d48036d
# ╠═2879f73e-9730-4d20-9035-b8c38001ccb8
# ╠═9eb6d679-3832-4297-a871-f2257a37602c
# ╠═3e35f741-12be-45d8-9a08-0a663824f0d5
# ╠═a1767185-78e0-47a4-bb17-59155b26b7f7
# ╠═c58ff9b5-6655-4994-8556-ee69ed3856b7
# ╠═79de0ef8-1658-4e2c-827d-3ec1f2f30378
# ╠═0fe68746-2ed3-42c6-babc-c231affe301f
# ╠═602c3751-9c8d-460a-b069-dfa5989a5cf5
# ╠═ca27253f-18c4-442d-bf29-e85ba3dadf3c
# ╠═69571f69-fdae-4de2-9a07-f2550e01a8ea
# ╠═cee050e0-f14d-4252-866a-8d399588f0a7
# ╠═ee4abdbc-bb3e-4414-b0e5-872c7ff5ee6f
# ╠═6e35d0c1-3bfa-415c-b747-2b6604e9f12a
# ╠═bb031fdc-35c7-4b97-96aa-aa0d05a34008
# ╠═ffa023d5-1798-4b85-9dd2-1e878f87c664
# ╠═f6450a10-4227-4a87-8703-e62014f22137
# ╠═0d199db9-9a9c-435f-b4aa-c4eaf8955cb7
