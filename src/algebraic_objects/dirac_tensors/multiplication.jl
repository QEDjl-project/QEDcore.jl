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
@inline function _mul(abs::AdjointBiSpinor, bs::BiSpinor)
    return abs.el1 * bs.el1 + abs.el2 * bs.el2 + abs.el3 * bs.el3 + abs.el4 * bs.el4
end
@inline Base.:*(abs::AdjointBiSpinor, bs::BiSpinor) =
    _mul(abs::AdjointBiSpinor, bs::BiSpinor)

"""
$(TYPEDSIGNATURES)

Tensor product of a standard with an adjoint bi-spinor resulting in a Dirac matrix.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
@inline function _mul(bs::BiSpinor, abs::AdjointBiSpinor)::DiracMatrix
    return return DiracMatrix(
        bs.el1 * abs.el1, bs.el2 * abs.el1, bs.el3 * abs.el1, bs.el4 * abs.el1,
        bs.el1 * abs.el2, bs.el2 * abs.el2, bs.el3 * abs.el2, bs.el4 * abs.el2,
        bs.el1 * abs.el3, bs.el2 * abs.el3, bs.el3 * abs.el3, bs.el4 * abs.el3,
        bs.el1 * abs.el4, bs.el2 * abs.el4, bs.el3 * abs.el4, bs.el4 * abs.el4,
    )
end
@inline Base.:*(bs::BiSpinor, abs::AdjointBiSpinor) =
    _mul(bs::BiSpinor, abs::AdjointBiSpinor)

"""
$(TYPEDSIGNATURES)

Tensor product of an Dirac matrix with a standard bi-spinor resulting in another standard bi-spinor.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
@inline function _mul(dm::DiracMatrix, bs::BiSpinor)
    return BiSpinor(
        dm.el11 * bs.el1 + dm.el21 * bs.el2 + dm.el31 * bs.el3 + dm.el41 * bs.el4,
        dm.el12 * bs.el1 + dm.el22 * bs.el2 + dm.el32 * bs.el3 + dm.el42 * bs.el4,
        dm.el13 * bs.el1 + dm.el23 * bs.el2 + dm.el33 * bs.el3 + dm.el43 * bs.el4,
        dm.el14 * bs.el1 + dm.el24 * bs.el2 + dm.el34 * bs.el3 + dm.el44 * bs.el4,
    )
end
@inline Base.:*(dm::DiracMatrix, bs::BiSpinor) = _mul(dm, bs)

"""
$(TYPEDSIGNATURES)

Tensor product of an adjoint bi-spinor with a Dirac matrix resulting in another adjoint bi-spinor.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
@inline function _mul(abs::AdjointBiSpinor, dm::DiracMatrix)
    return AdjointBiSpinor(
        abs.el1 * dm.el11 + abs.el2 * dm.el12 + abs.el3 * dm.el13 + abs.el4 * dm.el14,
        abs.el1 * dm.el21 + abs.el2 * dm.el22 + abs.el3 * dm.el23 + abs.el4 * dm.el24,
        abs.el1 * dm.el31 + abs.el2 * dm.el32 + abs.el3 * dm.el33 + abs.el4 * dm.el34,
        abs.el1 * dm.el41 + abs.el2 * dm.el42 + abs.el3 * dm.el43 + abs.el4 * dm.el44,
    )
end
@inline Base.:*(abs::AdjointBiSpinor, dm::DiracMatrix) = _mul(abs, dm)

"""
$(TYPEDSIGNATURES)

Tensor product two Dirac matrices resulting in another Dirac matrix.

!!! note "Multiplication operator"
    This also overloads the `*` operator for this types.
"""
@inline function _mul(dm1::DiracMatrix, dm2::DiracMatrix)::DiracMatrix
    return @inline DiracMatrix(SMatrix(dm1) * SMatrix(dm2))
end
@inline Base.:*(dm1::DiracMatrix, dm2::DiracMatrix)::DiracMatrix = _mul(dm1, dm2)

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
