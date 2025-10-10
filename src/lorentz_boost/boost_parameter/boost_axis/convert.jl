function Base.convert(
        ::Type{B}, param::Real
    ) where {T <: Real, B <: AbstractAxisBoostParameter{T}}
    return B(T(param))
end
function Base.convert(
        ::Type{B1}, d::B2
    ) where {T <: Real, B1 <: AbstractAxisBoostParameter{T}, B2 <: AbstractAxisBoostParameter}
    return B1(T(d.param))
end
function Base.convert(::Type{B}, d::B) where {B <: AbstractAxisBoostParameter}
    return d
end
