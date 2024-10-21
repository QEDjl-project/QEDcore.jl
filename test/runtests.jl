using QEDcore
using Test
using SafeTestsets

begin
    @time @safetestset "two body rest system" begin
        include("phase_space_layouts/in_channel/two_body/rest_system.jl")
    end

    @time @safetestset "Lorentz transform" begin
        include("lorentz_transform/lorentz_transform.jl")
    end

    # TODO: move this to QEDbase
    @time @safetestset "phase space layout" begin
        include("interfaces/phase_space_layout.jl")
    end

    @time @safetestset "coordinate map" begin
        include("coordinate_map.jl")
    end

    @time @safetestset "coordinates" begin
        include("coordinates.jl")
    end

    @time @safetestset "coordinate map" begin
        include("coordinate_map.jl")
    end

    @time @safetestset "phase spaces" begin
        include("phase_spaces.jl")
    end

    # algebraic objects
    @time @safetestset "four momentum" begin
        include("algebraic_objects/four_momentum.jl")
    end

    @time @safetestset "gamma matrices" begin
        include("algebraic_objects/gamma_matrices.jl")
    end

    @time @safetestset "Lorentz vector" begin
        include("algebraic_objects/lorentz_vector.jl")
    end

    @time @safetestset "Dirac tensors" begin
        include("algebraic_objects/dirac_tensor.jl")
    end

    # particles
    @time @safetestset "particle types" begin
        include("particles/types.jl")
    end

    @time @safetestset "particle states" begin
        include("particles/states.jl")
    end

    @time @safetestset "particle base states" begin
        include("particles/states.jl")
    end

    @time @safetestset "particle propagators" begin
        include("particles/propagators.jl")
    end

    # interfaces
    @time @safetestset "process interface" begin
        include("interfaces/process.jl")
    end
end
