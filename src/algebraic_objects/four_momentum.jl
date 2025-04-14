#######
#
# Concrete SFourMomentum type
#
#######

"""
$(TYPEDEF)

Builds a static LorentzVectorLike with real components used to statically model the four-momentum of a particle or field.

# Fields
$(TYPEDFIELDS)
"""
struct SFourMomentum{T <: Real} <: AbstractFourMomentum{T}
    "energy component"
    E::T

    "`x` component"
    px::T

    "`y` component"
    py::T

    "`z` component"
    pz::T

    """
    $(SIGNATURES)

    The generic constructor transforms each real-number input to the given `T_ELEM`:

    $(TYPEDSIGNATURES)
    """
    @inline function SFourMomentum{T_ELEM}(
            t::Real, x::Real, y::Real, z::Real
        ) where {T_ELEM <: Real}
        return new{T_ELEM}(T_ELEM(t), T_ELEM(x), T_ELEM(y), T_ELEM(z))
    end

    """
    $(SIGNATURES)

    The default constructor transforms each real-number input to `Float64`:

    $(TYPEDSIGNATURES)
    """
    @inline function SFourMomentum(t::Real, x::Real, y::Real, z::Real)
        return SFourMomentum{Float64}(t, x, y, z)
    end
end

function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{(4,)}
    ) where {A <: SFourMomentum, T <: Real}
    return SFourMomentum{T}
end
function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{(4,)}
    ) where {A <: SFourMomentum, T}
    return SLorentzVector{T}
end

@inline QEDbase.getT(p::SFourMomentum) = p.E
@inline QEDbase.getX(p::SFourMomentum) = p.px
@inline QEDbase.getY(p::SFourMomentum) = p.py
@inline QEDbase.getZ(p::SFourMomentum) = p.pz

# TODO: this breaks incremental compilation because it's trying to eval permanent changes in a different module
#register_LorentzVectorLike(SFourMomentum)
@traitimpl QEDbase.IsLorentzVectorLike{SFourMomentum}

#######
#
# Concrete MFourMomentum type
#
#######
"""
$(TYPEDEF)

Builds a mutable LorentzVector with real components used to statically model the four-momentum of a particle or field.

# Fields
$(TYPEDFIELDS)
"""
mutable struct MFourMomentum{T <: Real} <: AbstractFourMomentum{T}
    "energy component"
    E::T

    "`x` component"
    px::T

    "`y` component"
    py::T

    "`z` component"
    pz::T

    """
    $(SIGNATURES)

    The generic constructor transforms each real-number input to the given `T_ELEM`:

    $(TYPEDSIGNATURES)
    """
    @inline function MFourMomentum{T_ELEM}(
            t::Real, x::Real, y::Real, z::Real
        ) where {T_ELEM <: Real}
        return new{T_ELEM}(T_ELEM(t), T_ELEM(x), T_ELEM(y), T_ELEM(z))
    end

    """
    $(SIGNATURES)

    The default constructor transforms each real-number input to `Float64`:

    $(TYPEDSIGNATURES)
    """
    @inline function MFourMomentum(t::Real, x::Real, y::Real, z::Real)
        return MFourMomentum{Float64}(t, x, y, z)
    end
end

function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{(4,)}
    ) where {A <: MFourMomentum, T <: Real}
    return MFourMomentum{T}
end
function StaticArrays.similar_type(
        ::Type{A}, ::Type{T}, ::Size{(4,)}
    ) where {A <: MFourMomentum, T}
    return MLorentzVector{T}
end

@inline QEDbase.getT(p::MFourMomentum) = p.E
@inline QEDbase.getX(p::MFourMomentum) = p.px
@inline QEDbase.getY(p::MFourMomentum) = p.py
@inline QEDbase.getZ(p::MFourMomentum) = p.pz

function QEDbase.setT!(lv::MFourMomentum, value::Real)
    return lv.E = value
end

function QEDbase.setX!(lv::MFourMomentum, value::Real)
    return lv.px = value
end

function QEDbase.setY!(lv::MFourMomentum, value::Real)
    return lv.py = value
end

function QEDbase.setZ!(lv::MFourMomentum, value::Real)
    return lv.pz = value
end

# TODO: this breaks incremental compilation because it's trying to eval permanent changes in a different module
# register_LorentzVectorLike(MFourMomentum)
@traitimpl QEDbase.IsLorentzVectorLike{MFourMomentum}
@traitimpl QEDbase.IsMutableLorentzVectorLike{MFourMomentum}
