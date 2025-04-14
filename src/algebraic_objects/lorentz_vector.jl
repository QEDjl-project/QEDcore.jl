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

"""
$(TYPEDEF)

Concrete implementation of a generic mutable Lorentz vector. Each manipulation of an concrete implementation which is not self-contained (i.e. produces the same Lorentz vector type) will result in this type.

# Fields
$(TYPEDFIELDS)
"""
mutable struct MLorentzVector{T} <: AbstractLorentzVector{T}
    "`t` component"
    t::T

    "`x` component"
    x::T

    "`y` component"
    y::T

    "`z` component"
    z::T
end
MLorentzVector(t, x, y, z) = MLorentzVector(promote(t, x, y, z)...)

function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{S}
    ) where {A <: MLorentzVector, T, S}
    return MLorentzVector{T}
end

@inline QEDbase.getT(lv::MLorentzVector) = lv.t
@inline QEDbase.getX(lv::MLorentzVector) = lv.x
@inline QEDbase.getY(lv::MLorentzVector) = lv.y
@inline QEDbase.getZ(lv::MLorentzVector) = lv.z

function QEDbase.setT!(lv::MLorentzVector, value::T) where {T}
    return lv.t = value
end

function QEDbase.setX!(lv::MLorentzVector, value::T) where {T}
    return lv.x = value
end

function QEDbase.setY!(lv::MLorentzVector, value::T) where {T}
    return lv.y = value
end

function QEDbase.setZ!(lv::MLorentzVector, value::T) where {T}
    return lv.z = value
end

# TODO: this breaks incremental compilation because it's trying to eval permanent changes in a different module
#register_LorentzVectorLike(MLorentzVector)
@traitimpl QEDbase.IsLorentzVectorLike{MLorentzVector}
@traitimpl QEDbase.IsMutableLorentzVectorLike{MLorentzVector}
