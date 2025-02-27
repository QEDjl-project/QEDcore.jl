### Trivial phase-space layouts

# maps all components onto four momenta
struct TestInPhaseSpaceLayout <: AbstractInPhaseSpaceLayout end
Base.broadcastable(psl::TestInPhaseSpaceLayout) = Ref(psl)

@inline QEDbase.phase_space_dimension(
    proc::AbstractProcessDefinition, ::AbstractModelDefinition, ::TestInPhaseSpaceLayout
) = 4 * number_incoming_particles(proc)

# maps componets of N-1 particles onto four-momenta and uses energy-momentum conservation
struct TestOutPhaseSpaceLayout{INPSL} <: AbstractOutPhaseSpaceLayout{INPSL}
    in_psl::INPSL
end
Base.broadcastable(psl::TestOutPhaseSpaceLayout) = Ref(psl)

TestOutPhaseSpaceLayout() = TestOutPhaseSpaceLayout(TestInPhaseSpaceLayout())

@inline QEDbase.in_phase_space_layout(psl::TestOutPhaseSpaceLayout) = psl.in_psl

@inline QEDbase.phase_space_dimension(
    proc::AbstractProcessDefinition, ::AbstractModelDefinition, ::TestOutPhaseSpaceLayout
) = 4 * number_outgoing_particles(proc) - 4

struct TestInPhaseSpaceLayout_FAIL <: QEDbase.AbstractInPhaseSpaceLayout end
Base.broadcastable(psl::TestInPhaseSpaceLayout_FAIL) = Ref(psl)

struct TestOutPhaseSpaceLayout_FAIL <:
       QEDbase.AbstractOutPhaseSpaceLayout{TestInPhaseSpaceLayout}
    in_psl::TestInPhaseSpaceLayout
end
TestOutPhaseSpaceLayout_FAIL() = TestOutPhaseSpaceLayout_FAIL(TestInPhaseSpaceLayout())
Base.broadcastable(psl::TestOutPhaseSpaceLayout_FAIL) = Ref(psl)
