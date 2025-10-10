#######
#
# Abstract types
#
#######

#######
#
# Concrete LorentzVector types
#
#######
"""
$(TYPEDEF)

Concrete implementation of a generic static Lorentz vector. Each manipulation of an concrete implementation which is not self-contained (i.e. produces the same Lorentz vector type) will result in this type.

# Fields
$(TYPEDFIELDS)
"""
struct SLorentzVector{T} <: AbstractLorentzVector{T}
    "`t` component"
    t::T

    "`x` component"
    x::T

    "`y` component"
    y::T

    "`z` component"
    z::T
end
SLorentzVector(t, x, y, z) = SLorentzVector(promote(t, x, y, z)...)

function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{S}
    ) where {A <: SLorentzVector, T, S}
    return SLorentzVector{T}
end

@inline QEDbase.getT(lv::SLorentzVector) = lv.t
@inline QEDbase.getX(lv::SLorentzVector) = lv.x
@inline QEDbase.getY(lv::SLorentzVector) = lv.y
@inline QEDbase.getZ(lv::SLorentzVector) = lv.z

# TODO: this breaks incremental compilation because it's trying to eval permanent changes in a different module
#register_LorentzVectorLike(SLorentzVector)
@traitimpl QEDbase.IsLorentzVectorLike{SLorentzVector}
