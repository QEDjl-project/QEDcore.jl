abstract type AbstractTestModel <: AbstractModelDefinition end

struct TestModel <: AbstractTestModel end
QEDbase.fundamental_interaction_type(::TestModel) = :test_interaction

struct TestModel_FAIL <: AbstractModelDefinition end
