#######
#
# Concrete Dirac Tensor types
#
#######
"""
$(TYPEDEF)

Concrete type to model a Dirac four-spinor. These are the elements of an actual spinor space.
By default, a constructed `BiSpinor` will have complex-valued components, using `ComplexF64`, but
any other `Number` type can be used by explicitly calling `BiSpinor{T}(el1, el2, el3, el4)`, which converts
all given elements to `T`.
"""
struct BiSpinor{T<:Number} <: AbstractDiracVector{T}
    el1::T
    el2::T
    el3::T
    el4::T
end

"""
$(TYPEDEF)

Concrete type to model an adjoint Dirac four-spinor. These are the elements of the dual spinor space.
By default, a constructed `AdjointBiSpinor` will have complex-valued components, using `ComplexF64`, but
any other `Number` type can be used by explicitly calling `AdjointBiSpinor{T}(el1, el2, el3, el4)`, which converts
all given elements to `T`.
"""
struct AdjointBiSpinor{T<:Number} <: AbstractDiracVector{T}
    el1::T
    el2::T
    el3::T
    el4::T
end

AdjointBiSpinor(mat::AbstractVector{T}) where {T<:Number} = AdjointBiSpinor{T}(mat)
function similar_type(::Type{A}, ::Type{ElType}) where {A<:AdjointBiSpinor,ElType}
    return AdjointBiSpinor{ElType}
end
AdjointBiSpinor(sv::SVector{4,T}) where {T} = AdjointBiSpinor{T}(Tuple(sv))

#interface
AdjointBiSpinor(spn::BiSpinor{T}) where {T} = AdjointBiSpinor{T}(conj(SVector(spn)))
AdjointBiSpinor{T1}(spn::BiSpinor{T2}) where {T1, T2} = AdjointBiSpinor{promote_type(T1, T2)}(conj(SVector(spn)))
BiSpinor(spn::AdjointBiSpinor{T}) where {T} = BiSpinor{T}(conj(SVector(spn)))
BiSpinor{T1}(spn::AdjointBiSpinor{T2}) where {T1, T2} = BiSpinor{promote_type(T1, T2)}(conj(SVector(spn)))

"""
$(TYPEDEF)

Concrete type to model Dirac matrices, i.e. matrix representations of linear mappings between two spinor spaces.
"""
struct DiracMatrix{T<:Number} <: AbstractDiracMatrix{T}
    el11::T
    el12::T
    el13::T
    el14::T
    el21::T
    el22::T
    el23::T
    el24::T
    el31::T
    el32::T
    el33::T
    el34::T
    el41::T
    el42::T
    el43::T
    el44::T
end

DiracMatrix(mat::AbstractMatrix{T}) where {T<:Number} = DiracMatrix{T}(mat)
DiracMatrix(sm::SMatrix{4,4,T,16}) where {T} = DiracMatrix{T}(Tuple(sm))
