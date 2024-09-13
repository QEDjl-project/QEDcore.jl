
function Base.convert(
    ::Type{B}, param::S
) where {T<:Real,B<:AbstractAxisBoostParameter{T},S<:Real}
    return B(T(param))
end
function Base.convert(
    ::Type{B1}, d::B2
) where {T<:Real,B1<:AbstractAxisBoostParameter{T},B2<:AbstractAxisBoostParameter}
    return B1(T(d.param))
end
function Base.convert(
    ::Type{B1}, d::B2
) where {T<:Real,B1<:AbstractAxisBoostParameter{T},B2<:AbstractAxisBoostParameter{T}}
    return d
end
