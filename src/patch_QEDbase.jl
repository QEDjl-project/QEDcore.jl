# TODO:
# - QEDbase.jl: loose type specification to AbstractFourMomentum
# - QEDbase.jl: add easy constructor to MockInPhaseSpaceLayout
# - move this to QEDbase.jl
# - remove before merging!!!

# utility

@inline function split_uppercase(s::String)
    return split(s, r"(?=[A-Z])")
end

# general show for abstract particle types

function Base.show(io::IO, particle::T) where {T<:AbstractParticleType}
    t_string = string(T)
    lc_name = join(lowercase.(split_uppercase(t_string)), " ")

    print(io, "$(lc_name)")
    return nothing
end

# shows for other mocking types

function Base.show(io::IO, proc::QEDbase.Mocks.MockProcess)
    N = number_incoming_particles(proc)
    M = number_outgoing_particles(proc)
    print(io, "mock process ($N -> $M)")
    return nothing
end
Base.show(io::IO, m::QEDbase.Mocks.MockModel) = print(io, "mock model")

# think of a more sophisticated print
function Base.show(io::IO, psl::QEDbase.Mocks.MockOutPhaseSpaceLayout)
    return print(io, "mock out phase space layout")
end
