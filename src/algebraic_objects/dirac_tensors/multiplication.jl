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
    return DiracMatrix(BS * transpose(aBS))
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
    return transpose(aBS) * DM
end
@inline Base.:*(aBS::AdjointBiSpinor, DM::DiracMatrix) = _mul(aBS, DM)

"""
$(TYPEDSIGNATURES)

Tensor product two Dirac matrices resulting in another Dirac matrix.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.

"""
function _mul(DM1::DiracMatrix, DM2::DiracMatrix)::DiracMatrix
    return DM1 * DM2
end

"""
$(TYPEDSIGNATURES)

Tensor product of Dirac matrix sandwiched between an adjoint and a standard bi-spinor resulting in a scalar.
"""
function _mul(aBS::AdjointBiSpinor, DM::DiracMatrix, BS::BiSpinor)::ComplexF64
    return transpose(aBS) * DM * BS
end
