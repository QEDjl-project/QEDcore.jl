using QEDcore
using Test
using SafeTestsets

include("utils.jl")

# check if we run CPU tests (yes by default)
cpu_tests = _is_test_platform_active(["CI_QED_TEST_CPU", "TEST_CPU"], true)

if cpu_tests
    # miscellaneous utilities
    @time @safetestset "bisection" begin
        include("bisection.jl")
    end

    # main tests
    @time @safetestset "two body rest system" begin
        include("phase_space_layouts/in_channel/two_body/rest_system.jl")
    end

    @time @safetestset "two body heads-on system" begin
        include("phase_space_layouts/in_channel/two_body/heads_on.jl")
    end

    @time @safetestset "flat phase space layout" begin
        include("phase_space_layouts/out_channel/flat_phase_space.jl")
    end

    @time @safetestset "Lorentz transform" begin
        include("lorentz_transform/lorentz_transform.jl")
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
    @time @safetestset "constants" begin
        include("constants.jl")
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

    @time @safetestset "generic eltype tests" begin
        include("algebraic_objects/generic_eltypes.jl")
    end

    # particles
    @time @safetestset "particle types" begin
        include("particles/types.jl")
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
else
    @info "Skipping CPU tests"
end

begin
    @time @safetestset "GPU testing" begin
        include("gpu/runtests.jl")
    end
end
