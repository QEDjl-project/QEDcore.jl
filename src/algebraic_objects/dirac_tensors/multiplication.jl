#######
#
# Concrete implementation of multiplication for Dirac Tensors
#
#######

"""
$(TYPEDSIGNATURES)

Tensor product of an adjoint with a standard bi-spinor resulting in a scalar.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
function _mul(aBS::AdjointBiSpinor, BS::BiSpinor)
    return aBS.el1 * BS.el1 + aBS.el2 * BS.el2 + aBS.el3 * BS.el3 + aBS.el4 * BS.el4
end
@inline Base.:*(aBS::AdjointBiSpinor, BS::BiSpinor) =
    _mul(aBS::AdjointBiSpinor, BS::BiSpinor)

"""
$(TYPEDSIGNATURES)

Tensor product of a standard with an adjoint bi-spinor resulting in a Dirac matrix.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
function _mul(BS::BiSpinor, aBS::AdjointBiSpinor)::DiracMatrix
    return DiracMatrix(SVector(BS) * transpose(SVector(aBS)))
end
@inline Base.:*(BS::BiSpinor, aBS::AdjointBiSpinor) =
    _mul(BS::BiSpinor, aBS::AdjointBiSpinor)

"""
$(TYPEDSIGNATURES)

Tensor product of an Dirac matrix with a standard bi-spinor resulting in another standard bi-spinor.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
function _mul(DM::DiracMatrix, BS::BiSpinor)::BiSpinor
    return BiSpinor(SMatrix(DM) * SVector(BS))
end
@inline Base.:*(DM::DiracMatrix, BS::BiSpinor) = _mul(DM, BS)

"""
$(TYPEDSIGNATURES)

Tensor product of an adjoint bi-spinor with a Dirac matrix resulting in another adjoint bi-spinor.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
function _mul(aBS::AdjointBiSpinor, DM::DiracMatrix)::AdjointBiSpinor
    return AdjointBiSpinor(transpose(SVector(aBS)) * SMatrix(DM))
end
@inline Base.:*(aBS::AdjointBiSpinor, DM::DiracMatrix) = _mul(aBS, DM)

"""
$(TYPEDSIGNATURES)

Tensor product two Dirac matrices resulting in another Dirac matrix.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
function _mul(dm1::DiracMatrix, dm2::DiracMatrix)::DiracMatrix
    return DiracMatrix(SMatrix(dm1) * SMatrix(dm2))
end
@inline Base.:*(dm1::DiracMatrix, dm2::DiracMatrix)::DiracMatrix = _mul(dm1, dm2)

"""
$(TYPEDSIGNATURES)

Tensor product of Dirac matrix sandwiched between an adjoint and a standard bi-spinor resulting in a scalar.
"""
function _mul(abs::AdjointBiSpinor, dm::DiracMatrix, bs::BiSpinor)
    return transpose(SVector(abs)) * SMatrix(dm) * SVector(bs)
end
@inline Base.:*(abs::AdjointBiSpinor, dm::DiracMatrix, bs::BiSpinor) = _mul(abs, dm, bs)

"""
$(TYPEDSIGNATURES)

The product of a Dirac matrix with an adjoint bi-spinor from the right is not defined.
Therefore, this throws a method error.

!!! note
    We must throw this error explicitly, because otherwise the multiplication is
    dispatched to methods from StaticArrays.jl.
"""
function _mul(d::DiracMatrix, a::AdjointBiSpinor)
    throw(MethodError(*, (d, a)))
end
@inline Base.:*(d::DiracMatrix, a::AdjointBiSpinor) = _mul(d, a)

"""
$(TYPEDSIGNATURES)

The product of a Dirac matrix with a bi-spinor from the left is not defined.
Therefore, this throws a method error.

!!! note
    We must throw this error explicitly, because otherwise the multiplication is
    dispatched to methods from StaticArrays.jl.
"""
function _mul(b::BiSpinor, d::DiracMatrix)
    throw(MethodError(*, (b, d)))
end
@inline Base.:*(b::BiSpinor, d::DiracMatrix) = _mul(b, d)
