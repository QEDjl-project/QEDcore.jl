####
# concrete implementation of gamma matrices in Diracs representation
#
# Note: lower-index version of the gamma matrices are used
#       e.g. see https://en.wikipedia.org/wiki/Gamma_matrices
# Note: caused by the column major construction of matrices in Julia,
#       the definition below looks *transposed*.
####

function gamma(t::Type{T_GAMMA}) where {T<:Number,T_GAMMA<:AbstractGammaRepresentation{T}}
    return SLorentzVector(_gamma0(t), _gamma1(t), _gamma2(t), _gamma3(t))
end

struct DiracGammaRepresentation{T_ELEM<:Number} <: AbstractGammaRepresentation{T_ELEM} end

#! format: off
@inline function _gamma0(::Type{DiracGammaRepresentation{T_ELEM}})::DiracMatrix{T_ELEM} where {T_ELEM <: Number}
    return DiracMatrix{T_ELEM}(1,   0,   0,   0,
                               0,   1,   0,   0,
                               0,   0,  -1,   0,
                               0,   0,   0,  -1)
end

@inline function _gamma1(::Type{DiracGammaRepresentation{T_ELEM}})::DiracMatrix{T_ELEM} where {T_ELEM <: Number}
    return DiracMatrix{T_ELEM}(0,   0,   0,   1,
                               0,   0,   1,   0,
                               0,  -1,   0,   0,
                              -1,   0,   0,   0)
end

@inline function _gamma2(::Type{DiracGammaRepresentation{T_ELEM}})::DiracMatrix{T_ELEM} where {T_ELEM <: Number}
    return DiracMatrix{T_ELEM}(0,    0,    0, 1im,
                               0,    0, -1im,   0,
                               0, -1im,    0,   0,
                               1im,  0,    0,   0)
end

@inline function _gamma3(::Type{DiracGammaRepresentation{T_ELEM}})::DiracMatrix{T_ELEM} where {T_ELEM <: Number}
    return DiracMatrix{T_ELEM}(0,  0,  1,  0,
                               0,  0,  0, -1,
                              -1,  0,  0,  0,
                               0,  1,  0,  0)
end
#! format: on

# default gamma matrix is in Dirac's representation
@inline gamma() = gamma(DiracGammaRepresentation{ComplexF64})
@inline gamma(::Type{T_ELEM}) where {T_ELEM<:Number} =
    gamma(DiracGammaRepresentation{T_ELEM})
@inline gamma(::Type{<:Real}) =
    throw(ArgumentError("cannot create a non-complex-valued gamma matrix\n"))

@inline _complex_from_real_t(::Type{T_ELEM}) where {T_ELEM<:Real} = ComplexF64
@inline _complex_from_real_t(::Type{Float64}) = ComplexF64
@inline _complex_from_real_t(::Type{Float32}) = ComplexF32
@inline _complex_from_real_t(::Type{Float16}) = ComplexF16

# feynman slash notation

function slashed(
    ::Type{T_GAMMA}, lv::AbstractLorentzVector{T_ELEM}
) where {T_ELEM<:Number,T_GAMMA<:AbstractGammaRepresentation{T_ELEM}}
    return gamma(T_GAMMA) * lv
end

function slashed(LV::T) where {T_ELEM<:Real,T<:AbstractLorentzVector{T_ELEM}}
    return gamma(_complex_from_real_t(T_ELEM)) * LV
end

function slashed(LV::T) where {T_ELEM<:Number,T<:AbstractLorentzVector{T_ELEM}}
    return gamma(T_ELEM) * LV
end
