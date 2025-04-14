abstract type AbstractTestModel <: AbstractModelDefinition end

struct TestModel <: AbstractTestModel end
QEDbase.fundamental_interaction_type(::TestModel) = :test_interaction

struct TestPerturbativeModel <: QEDcore.AbstractPerturbativeModel end

struct TestModel_FAIL <: AbstractModelDefinition end

# for dispatch
const TestModels = Union{TestModel, TestPerturbativeModel}
