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

# ╔═╡ 31ad4bc9-d22b-41e6-9d70-6defaea25ae9
begin
    Pkg.add("QEDprocesses")
    using QEDprocesses
end

# ╔═╡ 66ac2e14-89f3-11ef-33e5-7ba07de31814
md"# _Phase Space Layout_"

# ╔═╡ 2aac4d0b-f197-46ea-85fb-06aa324d4e22
md"## Test implementation: in PSL"

# ╔═╡ a8279fc1-ccd0-4246-9d9e-d087195ecbaf
begin
    struct TwoBodyRestSystem{RESTIDX,COORD<:AbstractUnivariateCoordinates} <:
           AbstractInPhaseSpaceLayout
        coord::COORD

        function TwoBodyRestSystem{RESTIDX}(
            coord_name::COORD
        ) where {RESTIDX,COORD<:AbstractUnivariateCoordinates}
            # TODO: add validity check
            # - only particle idx != RESTIDX are allowed
            # - only Energy, CMSEnergy, Rapidity and SpatialMagnitude are allowed
            return new{RESTIDX,COORD}(coord_name)
        end
    end

    @inline TwoBodyRestSystem(::Val{IDX}, coord_name) where {IDX} =
        TwoBodyRestSystem{IDX}(coord_name)
    @inline TwoBodyRestSystem(idx::Int, coord_name) =
        TwoBodyRestSystem(Val(idx), coord_name)
    TwoBodyRestSystem() = TwoBodyRestSystem(1, Energy(2))
    const TwoBodyTargetSystem{COORD} =
        TwoBodyRestSystem{1,COORD} where {COORD<:AbstractUnivariateCoordinates}
    TwoBodyTargetSystem() = TwoBodyTargetSystem(Energy(2))

    function QEDcore.phase_space_dimension(
        proc::Compton, model::PerturbativeQED, in_psl::TwoBodyRestSystem
    )
        return 1
    end

    #=
        function QEDcore._build_momenta(
            proc::AbstractProcessDefinition, ::PerturbativeQED, ::TwoBodyTargetSystem{<:Energy}, in_coords
        )
    		mass1 , mass2 = mass.(incoming_particles(proc))
            P1 = SFourMomentum(mass1, 0, 0, 0)

            energy2 = @inbounds in_coords[1]
    		rho2 = sqrt(energy2^2 - mass2^2)
            P2 = SFourMomentum(energy2, 0, 0, rho2)
            return P1, P2
        end
    =#

    select_moms(::Val{1}, ::Val{2}, P_rest, P_run) = (P_rest, P_run)
    select_moms(::Val{2}, ::Val{1}, P_rest, P_run) = (P_run, P_rest)
    function select_moms(rest_idx::Int, run_idx::Int, P_rest, P_run)
        return select_moms(Val(rest_idx), Val(run_idx), P_rest, P_run)
    end

    function QEDcore._build_momenta(
        proc::AbstractProcessDefinition,
        ::PerturbativeQED,
        ::TwoBodyRestSystem{RESTIDX,<:Energy{RUNIDX}},
        in_coords,
    ) where {RESTIDX,RUNIDX}
        masses = mass.(incoming_particles(proc))
        mass_rest = masses[RESTIDX]
        P_rest = SFourMomentum(mass_rest, 0, 0, 0)

        mass_run = masses[RUNIDX]
        energy_run = @inbounds in_coords[1]

        rho_run = sqrt(energy_run^2 - mass_run^2)
        P_run = SFourMomentum(energy_run, 0, 0, rho_run)

        return select_moms(RESTIDX, RUNIDX, P_rest, P_run)
    end

    function QEDcore._build_momenta(
        proc::AbstractProcessDefinition,
        ::PerturbativeQED,
        ::TwoBodyTargetSystem{<:CMSEnergy},
        in_coords,
    )
        mass1, mass2 = mass.(incoming_particles(proc))
        P1 = SFourMomentum(mass1, 0, 0, 0)

        sqrt_s = @inbounds in_coords[1]
        energy2 = (sqrt_s^2 - mass1^2 - mass2^2) / (2 * mass1)
        rho2 = sqrt(energy2^2 - mass2^2)
        P2 = SFourMomentum(energy2, 0, 0, rho2)
        return P1, P2
    end
end

# ╔═╡ 1af28cf1-15ab-4754-950d-58d3b56514b6
begin
    PROC = Compton()
    MODEL = PerturbativeQED()
    INPSL = TwoBodyTargetSystem()
    IN_COORDS = (10.0,)

    in_moms = build_momenta(PROC, MODEL, INPSL, IN_COORDS)
end

# ╔═╡ 17c95263-014e-4183-9024-3e52354d607f
@code_lowered build_momenta(PROC, MODEL, INPSL, IN_COORDS)

# ╔═╡ a0a64e53-a5ae-470a-86fb-aedf11ba1db1
@benchmark build_momenta($PROC, $MODEL, $INPSL, $IN_COORDS)

# ╔═╡ 0f6af71f-318f-43f6-bf20-ac6e7e673203
begin
    p_tot = sum(in_moms)
    s = p_tot * p_tot
    ss = sqrt(s)

    INPSL_SS = TwoBodyTargetSystem(CMSEnergy())
    #@show QEDcore.coordinate_names(INPSL_SS)
    IN_COORDS_SS = (ss,)
    build_momenta(PROC, MODEL, INPSL_SS, IN_COORDS_SS)
end

# ╔═╡ f995fc21-6238-49c9-bd5c-ae7da14103d0
@benchmark build_momenta($PROC, $MODEL, $INPSL_SS, $IN_COORDS_SS)

# ╔═╡ 62f784c6-d8c9-4a7e-ab0c-540846a77a0e
md"## Test implementation: out psl"

# ╔═╡ b41ed8cf-d4f2-4c14-8ac3-cbbbd9bc9f3d
begin
    struct tSphericalCoords{INPSL} <: AbstractOutPhaseSpaceLayout{INPSL}
        in_psl::INPSL
    end

    QEDcore.in_phase_space_layout(out_psl::tSphericalCoords) = out_psl.in_psl

    QEDcore.phase_space_dimension(proc, model, ::tSphericalCoords) = 2

    function QEDcore._build_momenta(
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

# ╔═╡ 02ba9413-0c77-4201-869c-400dc56c162c
begin
    OUTPSL = tSphericalCoords(INPSL)

    IN_MOMS = build_momenta(PROC, MODEL, INPSL, IN_COORDS)
    PTOT = sum(IN_MOMS)

    out_coords = (0.9, 0.1)
    OUT_MOMS = build_momenta(PROC, MODEL, PTOT, OUTPSL, out_coords)
end

# ╔═╡ 657e4f88-633a-493d-b8c1-931b593e1b23
sum(IN_MOMS) - sum(OUT_MOMS)

# ╔═╡ bd2f4325-75b5-4c42-94da-759a6d91860d
@benchmark build_momenta($PROC, $MODEL, $PTOT, $OUTPSL, $out_coords)

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
e = Energy(1)

# ╔═╡ a0e00255-f08e-40a6-a476-c17f45c8f3e0
function test(proc, c)
    idx = QEDcore.particle_index(c)
    in_parts = incoming_particles(proc)
    return mass(in_parts[idx])
end

# ╔═╡ c0cc204d-09f3-4402-805f-fc49d3694c6b
@code_llvm test(Compton(), e)

# ╔═╡ Cell order:
# ╠═66ac2e14-89f3-11ef-33e5-7ba07de31814
# ╠═f190fea0-3bf3-4b67-a3cc-42a812cf96c2
# ╠═31ad4bc9-d22b-41e6-9d70-6defaea25ae9
# ╠═2aac4d0b-f197-46ea-85fb-06aa324d4e22
# ╠═a8279fc1-ccd0-4246-9d9e-d087195ecbaf
# ╠═1af28cf1-15ab-4754-950d-58d3b56514b6
# ╠═17c95263-014e-4183-9024-3e52354d607f
# ╠═a0a64e53-a5ae-470a-86fb-aedf11ba1db1
# ╠═0f6af71f-318f-43f6-bf20-ac6e7e673203
# ╠═f995fc21-6238-49c9-bd5c-ae7da14103d0
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
# ╠═a0e00255-f08e-40a6-a476-c17f45c8f3e0
# ╠═c0cc204d-09f3-4402-805f-fc49d3694c6b
