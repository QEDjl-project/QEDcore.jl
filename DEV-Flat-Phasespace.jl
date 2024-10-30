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
end

# ╔═╡ 3fd1f0e2-7733-4306-8fd6-6d5edb5afd1b
in_psl = TwoBodyRestSystem(1, Energy(2))

# ╔═╡ 863dc644-8731-45fe-aa71-1ca3adbc0e3d
PROC = ScatteringProcess((Electron(), Positron()), (Electron(), Positron()))

# ╔═╡ c3086d07-b18f-41c8-99b6-06ec6b909b95
struct pQED <: QEDcore.AbstractPerturbativeModel end

# ╔═╡ 85c4dd62-1b6a-40e6-b793-626538fe6e39
MODEL = pQED()

# ╔═╡ 0dbbeec8-8bdd-4ec8-b8b9-4c006fe552b9
coord_map = CoordinateMap(PROC, MODEL, in_psl)

# ╔═╡ eb050099-0431-44ce-b6b1-5645fbd3d17a
coord_map((10,))

# ╔═╡ 482a5693-608b-480d-9423-ee3b43533c4a
md"# Current implementation"

# ╔═╡ 881c62f1-4548-4bb2-9dab-ff761014effa
begin
    function _uni2mom(u1, u2, u3, u4)
        cth = 2 * u1 - 1
        sth = sqrt(1 - cth^2)
        phi = 2 * pi * u2
        q0 = -log(u3 * u4)
        qx = q0 * sth * cos(phi)
        qy = q0 * sth * sin(phi)
        qz = q0 * cth

        return SFourMomentum(q0, qx, qy, qz)
    end

    _uni2mom(u1234::Tuple) = _uni2mom(u1234...)
    _uni2mom(u1234::SFourMomentum) = _uni2mom(Tuple(u1234))

    # generation of rambo momenta (without energy-momentum conservation)
    function generate_rambo_moms(rng, n::Int)
        a = Vector{SFourMomentum}(undef, n)
        rand!(rng, a)
        return map(_uni2mom, a)
    end

    # generation of physical massless momenta (with energy-momentum conservation)
    function _physical_momenta(coords::Tuple, ss::Real)
        r_moms = _uni2mom.(map(SFourMomentum, Iterators.partition(coords, 4)))
        n = length(r_moms)
        Q = sum(r_moms)
        M = sqrt(Q * Q)
        fac = -1 / M
        Qx = getX(Q)
        Qy = getY(Q)
        Qz = getZ(Q)
        bx = fac * Qx
        by = fac * Qy
        bz = fac * Qz
        gamma = getT(Q) / M
        a = 1 / (1 + gamma)
        x = ss / M

        i = 1
        while i <= n
            mom = r_moms[i]
            mom0 = getT(mom)
            mom1 = getX(mom)
            mom2 = getY(mom)
            mom3 = getZ(mom)

            bq = bx * mom1 + by * mom2 + bz * mom3

            p0 = x * (gamma * mom0 + bq)
            px = x * (mom1 + bx * mom0 + a * bq * bx)
            py = x * (mom2 + by * mom0 + a * bq * by)
            pz = x * (mom3 + bz * mom0 + a * bq * bz)

            r_moms[i] = SFourMomentum(p0, px, py, pz)
            i += 1
        end
        return r_moms
    end

    # generation of physical massless momenta (with energy-momentum conservation)
    function generate_physical_momenta(rng, n, ss)
        rnd_coords = Tuple(rand(rng, 4 * n))
        return _physical_momenta(rnd_coords, ss)
    end

    function _to_be_solved(xi, masses, p0s, ss)
        sum = 0.0
        for (i, E) in enumerate(p0s)
            sum += sqrt(masses[i]^2 + xi^2 * E^2)
        end
        return sum - ss
    end

    function _build_massive_momenta(xi, masses, massless_moms)
        vec = SFourMomentum[]
        i = 1
        while i <= length(massless_moms)
            massless_mom = massless_moms[i]
            k0 = sqrt(getT(massless_mom)^2 * xi^2 + masses[i]^2)

            kx = xi * getX(massless_mom)
            ky = xi * getY(massless_mom)
            kz = xi * getZ(massless_mom)

            push!(vec, SFourMomentum(k0, kx, ky, kz))

            i += 1
        end
        return vec
    end

    first_derivative(func) = x -> ForwardDiff.derivative(func, float(x))

    function generate_physical_massive_moms(rng, ss, masses; x0=0.1)
        n = length(masses)
        massless_moms = generate_physical_momenta(rng, n, ss)
        energies = getT.(massless_moms)
        f = x -> _to_be_solved(x, masses, energies, ss)
        #xi = find_zero((f, first_derivative(f)), ss, Roots.Newton())
        xi = find_zero(f, (1e-12, 2), Roots.Bisection())
        return _build_massive_momenta(xi, masses, massless_moms)
    end
end

# ╔═╡ 50129475-77f9-4199-8005-22a596a1cdb4
begin
    E = 2

    IN_MOMS = (
        SFourMomentum(E, 0, 0, sqrt(E^2 - 1)), SFourMomentum(E, 0, 0, -sqrt(E^2 - 1))
    )
    SS = getMass(sum(IN_MOMS))

    RND_COORDS = Tuple(rand(4 * 4))

    phys_mom = _physical_momenta(RND_COORDS, SS)

    energies = getT.(phys_mom)

    #m = [20.0,100.0,0.0,10.0]
    m = [0.01, 0.0, 0.0, 1.0]

    plot(x -> _to_be_solved(x, m, energies, SS))
end

# ╔═╡ 356b3335-ef88-4b42-8eff-54fe31266dba
begin
    RNG = MersenneTwister(1)
    generate_physical_massive_moms(RNG, SS, m)
end

# ╔═╡ e1c787d0-86f1-4ca9-b118-456e3311e47b
@benchmark _physical_momenta($RND_COORDS, $SS)

# ╔═╡ 26b82719-caf5-4d24-98c3-e4f7b063ce0f
@code_warntype _physical_momenta(RND_COORDS, SS)

# ╔═╡ Cell order:
# ╠═219d5c8e-96c4-11ef-350b-691d878780a3
# ╠═3fd1f0e2-7733-4306-8fd6-6d5edb5afd1b
# ╠═863dc644-8731-45fe-aa71-1ca3adbc0e3d
# ╠═c3086d07-b18f-41c8-99b6-06ec6b909b95
# ╠═85c4dd62-1b6a-40e6-b793-626538fe6e39
# ╠═0dbbeec8-8bdd-4ec8-b8b9-4c006fe552b9
# ╠═eb050099-0431-44ce-b6b1-5645fbd3d17a
# ╟─482a5693-608b-480d-9423-ee3b43533c4a
# ╠═881c62f1-4548-4bb2-9dab-ff761014effa
# ╠═356b3335-ef88-4b42-8eff-54fe31266dba
# ╠═50129475-77f9-4199-8005-22a596a1cdb4
# ╠═e1c787d0-86f1-4ca9-b118-456e3311e47b
# ╠═26b82719-caf5-4d24-98c3-e4f7b063ce0f
