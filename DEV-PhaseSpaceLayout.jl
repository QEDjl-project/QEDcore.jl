### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ f190fea0-3bf3-4b67-a3cc-42a812cf96c2
begin
    using Pkg: Pkg
    Pkg.activate(".")

    using QEDbase
    using QEDcore
    using BenchmarkTools
end

# ╔═╡ 66ac2e14-89f3-11ef-33e5-7ba07de31814
md"# _Phase Space Layout_"

# ╔═╡ 79d9350a-83e6-4463-8625-bb745c37ec35
md"## Abstract interface"

# ╔═╡ 7b298057-3701-4ec8-bbf6-7e578b4bf9d4
begin
    abstract type AbstractPhaseSpaceLayout end

    abstract type AbstractInPhaseSpaceLayout <: AbstractPhaseSpaceLayout end

    abstract type AbstractOutPhaseSpaceLayout{InPSL<:AbstractInPhaseSpaceLayout} <:
                  AbstractPhaseSpaceLayout end

    """
    	ps_dim(proc,model,::AbstractPhaseSpaceLayout)
    """
    function ps_dim end

    """
    	in_phase_space_layout(::AbstactOutPhaseSpaceLayout)
    """
    function in_phase_space_layout end

    """
    	_unsafe_build_momenta(proc,model,in_psl, in_coords)
    	_unsafe_build_momenta(proc,model,Ptot,out_psl,out_coords)
    """
    function _unsafe_build_momenta end

    """
    	is_valid(proc,model,in_psl,in_coords)
    	is_valid(proc,model,Ptot,out_psl,out_coords)
    """
    function is_valid end
end

# ╔═╡ 74a8230f-afa6-400b-9555-c58637b67b6f
md"## Derived functions: in psl"

# ╔═╡ 2aac4d0b-f197-46ea-85fb-06aa324d4e22
md"## Test implementation: in PSL"

# ╔═╡ a8279fc1-ccd0-4246-9d9e-d087195ecbaf
begin
    struct tElectronRestSystem <: AbstractInPhaseSpaceLayout end

    ps_dim(proc::Compton, model::PerturbativeQED, in_psl::tElectronRestSystem) = 1

    function _unsafe_build_momenta(
        ::Compton, ::PerturbativeQED, ::tElectronRestSystem, in_coords
    )
        P_electron = SFourMomentum(1, 0, 0, 0)

        omega = @inbounds in_coords[1]
        K_photon = SFourMomentum(omega, 0, 0, omega)
        return P_electron, K_photon
    end
end

# ╔═╡ 9bb5a4a9-90d7-43e7-aed1-e773e5ea3137
md"## Derived functions: out psl"

# ╔═╡ 62f784c6-d8c9-4a7e-ab0c-540846a77a0e
md"## Test implementation: out psl"

# ╔═╡ b41ed8cf-d4f2-4c14-8ac3-cbbbd9bc9f3d
begin
    struct tSphericalCoords{INPSL} <: AbstractOutPhaseSpaceLayout{INPSL}
        in_psl::INPSL
    end

    in_phase_space_layout(out_psl::tSphericalCoords) = out_psl.in_psl

    ps_dim(proc, model, ::tSphericalCoords) = 2

    function _unsafe_build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        Ptot::AbstractFourMomentum,
        out_psl::tSphericalCoords,
        out_coords,
    )
        cth, phi = out_coords
        s = getMass(Ptot)
        rho_tot = getMagnitude(Ptot)

        omp = (s - one(s)) / (getEnergy(Ptot) - rho_tot * cth)

        sth = sqrt(1 - cth^2)
        sphi, cphi = sincos(phi)

        K_prime = SFourMomentum(omp, omp * sth * cphi, omp * sth * sphi, omp * cth)
        P_prime = Ptot - K_prime

        return (P_prime, K_prime)
    end
end

# ╔═╡ edf84e69-51bb-4d67-a7d0-dce4326c7856
begin
    """
    	Safe version of _unsafe_build_momenta for in_psl

    !!! note

    	- use generic coord container

    """
    function _build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        in_psl::AbstractInPhaseSpaceLayout,
        in_coords,
    )
        length(in_coords) == ps_dim(proc, model, in_psl) ||
            throw(InvalidInputError("Wrong dimension!"))

        return _unsafe_build_momenta(proc, model, in_psl, in_coords)
    end

    function _build_momenta(
        ::Val{Nc}, proc, model, in_psl, in_coords::NTuple{N}
    ) where {Nc,N}
        throw(InvalidInputError("Wrong dimension!"))
    end

    function _build_momenta(
        ::Val{Nc},
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        in_psl::AbstractInPhaseSpaceLayout,
        in_coords::NTuple{Nc,T},
    ) where {Nc,T}
        return _unsafe_build_momenta(proc, model, in_psl, in_coords)
    end

    function build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        in_psl::AbstractInPhaseSpaceLayout,
        in_coords,
    )
        return _build_momenta(
            Val(ps_dim(proc, model, in_psl)), proc, model, in_psl, in_coords
        )
    end

    """
    	Scalar version of _build_momenta for in_psl
    """
    function _build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        in_psl::AbstractInPhaseSpaceLayout,
        in_coords::Real,
    )
        return _build_momenta(proc, model, in_psl, (in_coords,))
    end
end

# ╔═╡ 1af28cf1-15ab-4754-950d-58d3b56514b6
begin
    PROC = Compton()
    MODEL = PerturbativeQED()
    INPSL = tElectronRestSystem()
    IN_COORDS = (10.0,)

    @code_llvm build_momenta(PROC, MODEL, INPSL, IN_COORDS)
end

# ╔═╡ a0a64e53-a5ae-470a-86fb-aedf11ba1db1
@benchmark build_momenta($PROC, $MODEL, $INPSL, $IN_COORDS)

# ╔═╡ e325b727-8b9d-4bb9-9c0b-2543cc3a9218
@benchmark _unsafe_build_momenta($PROC, $MODEL, $INPSL, $IN_COORDS)

# ╔═╡ 7e9dd67b-1e00-4949-8d2d-40029ceb42be
begin
    """
   	Safe version of _unsafe_build_momenta for out_psl

   !!! note

   	- use generic coord container

   """
    function _build_momenta(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        Ptot::AbstractFourMomentum,
        out_psl::AbstractOutPhaseSpaceLayout,
        out_coords,
    )
        length(out_coords) == ps_dim(proc, model, out_psl) ||
            throw(InvalidInputError("Wrong dimension!"))

        return _unsafe_build_momenta(proc, model, Ptot, out_psl, out_coords)
    end
end

# ╔═╡ fcae4171-baca-43b4-a033-2ffe73f27e55
@benchmark _build_momenta($PROC, $MODEL, $INPSL, $IN_COORDS)

# ╔═╡ 02ba9413-0c77-4201-869c-400dc56c162c
begin
    OUTPSL = tSphericalCoords(INPSL)

    IN_MOMS = _build_momenta(PROC, MODEL, INPSL, IN_COORDS)
    PTOT = sum(IN_MOMS)

    out_coords = (0.9, 0.1)
    OUT_MOMS = _build_momenta(PROC, MODEL, PTOT, OUTPSL, out_coords)
end

# ╔═╡ 657e4f88-633a-493d-b8c1-931b593e1b23
sum(IN_MOMS) - sum(OUT_MOMS)

# ╔═╡ bd2f4325-75b5-4c42-94da-759a6d91860d
@benchmark _build_momenta($PROC, $MODEL, $PTOT, $OUTPSL, $out_coords)

# ╔═╡ a6e39a25-9aec-411b-a7e7-0f93de88e599
md"## Coordinate Map"

# ╔═╡ 18212ae3-0fbd-4ee8-9a4e-c7c83d6548b1
begin
    abstract type AbstractCoordianteMap end

    struct CoordinateMap{P,M,PSL<:AbstractPhaseSpaceLayout} <: AbstractCoordianteMap
        proc::P
        model::M
        psl::PSL

        # TODO: validity check
        # - compatibility of proc, model, psl
    end

    # make the transform callable
    @inline function (coord_map::CoordinateMap{P,M,PSL})(
        in_coords::Tuple
    ) where {P,M,PSL<:AbstractInPhaseSpaceLayout}
        return build_momenta(coord_map.proc, coord_map.model, coord_map.psl, in_coords)
    end

    # make the transform callable
    @inline function (coord_map::CoordinateMap{P,M,PSL})(
        in_coords::Tuple, out_coords::Tuple
    ) where {P,M,PSL<:AbstractOutPhaseSpaceLayout}
        in_moms = build_momenta(
            coord_map.proc, coord_map.model, in_phase_space_layout(coord_map.psl), in_coords
        )
        Ptot = sum(in_moms)
        return in_moms,
        _unsafe_build_momenta(
            coord_map.proc, coord_map.model, Ptot, coord_map.psl, out_coords
        )
    end
end

# ╔═╡ 8fed6c06-64d6-49b3-aecf-2435a2e475ce
cmap = CoordinateMap(PROC, MODEL, INPSL)

# ╔═╡ 7b8e27f6-0a3c-4db5-a6a9-f516c4369c83
@benchmark $cmap($IN_COORDS)

# ╔═╡ ac3e3b60-38f2-4f08-bcf4-a063d6c15bb8
cmap2 = CoordinateMap(PROC, MODEL, OUTPSL)

# ╔═╡ 61558d2a-8a33-4fff-b678-5a334f9f89d7
@benchmark $cmap2($IN_COORDS, $out_coords)

# ╔═╡ 34c3a028-e199-4fec-b96b-f63a36b538d3
md"## Coordinate map (cached)"

# ╔═╡ 89f245e0-9f49-4a86-9236-b0a32ed825e5
begin
    struct CoordinateMapCached{P,M,PSL<:AbstractPhaseSpaceLayout,TM<:Tuple} <:
           AbstractCoordianteMap
        proc::P
        model::M
        psl::PSL
        in_moms::TM
    end

    function CoordinateMapCached(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractInPhaseSpaceLayout,
        in_coords::NTuple{N,T},
    ) where {N,T<:Real}
        in_moms = build_momenta(proc, model, psl, in_coords)
        return CoordinateMapCached(proc, model, psl, in_moms)
    end

    function CoordinateMapCached(
        proc::AbstractProcessDefinition,
        model::AbstractModelDefinition,
        psl::AbstractOutPhaseSpaceLayout,
        in_coords::NTuple{N,T},
    ) where {N,T<:Real}
        in_moms = build_momenta(proc, model, in_phase_space_layout(psl), in_coords)
        return CoordinateMapCached(proc, model, psl, in_moms)
    end

    # make the transform callable
    @inline function (
        coord_map::CoordinateMapCached{P,M,PSL}
    )() where {P,M,PSL<:AbstractInPhaseSpaceLayout}
        return getfield(coord_map, :in_moms)
    end

    # make the transform callable
    @inline function (coord_map::CoordinateMapCached{P,M,PSL})(
        out_coords::Tuple; ret_in_moms=false
    ) where {P,M,PSL<:AbstractOutPhaseSpaceLayout}
        in_moms = coord_map.in_moms
        Ptot = sum(in_moms)
        return _unsafe_build_momenta(
            coord_map.proc, coord_map.model, Ptot, coord_map.psl, out_coords
        )
    end
end

# ╔═╡ b3fc817c-d36c-46dd-b66f-07fce460a8ea
ccmap = CoordinateMapCached(PROC, MODEL, INPSL, IN_COORDS)

# ╔═╡ a3fe4d97-78c0-4566-853d-423a8191174a
@benchmark CoordinateMapCached($PROC, $MODEL, $INPSL, $IN_COORDS)

# ╔═╡ 6d5aea4a-39bc-40b5-ae0e-ba2ff9211c7d
@benchmark $ccmap()

# ╔═╡ 81c19df5-b475-4a30-bda7-aa0a5e820c62
ccmap2 = CoordinateMapCached(PROC, MODEL, OUTPSL, IN_COORDS)

# ╔═╡ 6f3efdf6-332c-4e76-b4fb-3961afd2cd4f
@benchmark $ccmap2($out_coords)

# ╔═╡ 6cdc123a-44ad-49c7-8c8a-e3edf6c55cc6
@benchmark sum($IN_MOMS)

# ╔═╡ b9ff9df4-ca24-445c-aace-f974748bd206

# ╔═╡ Cell order:
# ╠═66ac2e14-89f3-11ef-33e5-7ba07de31814
# ╠═f190fea0-3bf3-4b67-a3cc-42a812cf96c2
# ╠═79d9350a-83e6-4463-8625-bb745c37ec35
# ╠═7b298057-3701-4ec8-bbf6-7e578b4bf9d4
# ╠═74a8230f-afa6-400b-9555-c58637b67b6f
# ╠═edf84e69-51bb-4d67-a7d0-dce4326c7856
# ╠═2aac4d0b-f197-46ea-85fb-06aa324d4e22
# ╠═a8279fc1-ccd0-4246-9d9e-d087195ecbaf
# ╠═1af28cf1-15ab-4754-950d-58d3b56514b6
# ╠═e325b727-8b9d-4bb9-9c0b-2543cc3a9218
# ╠═fcae4171-baca-43b4-a033-2ffe73f27e55
# ╠═a0a64e53-a5ae-470a-86fb-aedf11ba1db1
# ╠═9bb5a4a9-90d7-43e7-aed1-e773e5ea3137
# ╠═7e9dd67b-1e00-4949-8d2d-40029ceb42be
# ╠═62f784c6-d8c9-4a7e-ab0c-540846a77a0e
# ╠═b41ed8cf-d4f2-4c14-8ac3-cbbbd9bc9f3d
# ╠═02ba9413-0c77-4201-869c-400dc56c162c
# ╠═657e4f88-633a-493d-b8c1-931b593e1b23
# ╠═bd2f4325-75b5-4c42-94da-759a6d91860d
# ╟─a6e39a25-9aec-411b-a7e7-0f93de88e599
# ╠═18212ae3-0fbd-4ee8-9a4e-c7c83d6548b1
# ╠═8fed6c06-64d6-49b3-aecf-2435a2e475ce
# ╠═7b8e27f6-0a3c-4db5-a6a9-f516c4369c83
# ╠═ac3e3b60-38f2-4f08-bcf4-a063d6c15bb8
# ╠═61558d2a-8a33-4fff-b678-5a334f9f89d7
# ╠═34c3a028-e199-4fec-b96b-f63a36b538d3
# ╠═89f245e0-9f49-4a86-9236-b0a32ed825e5
# ╠═b3fc817c-d36c-46dd-b66f-07fce460a8ea
# ╠═a3fe4d97-78c0-4566-853d-423a8191174a
# ╠═6d5aea4a-39bc-40b5-ae0e-ba2ff9211c7d
# ╠═81c19df5-b475-4a30-bda7-aa0a5e820c62
# ╠═6f3efdf6-332c-4e76-b4fb-3961afd2cd4f
# ╠═6cdc123a-44ad-49c7-8c8a-e3edf6c55cc6
# ╠═b9ff9df4-ca24-445c-aace-f974748bd206
