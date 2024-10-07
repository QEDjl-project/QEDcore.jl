
function Base.isapprox(
    b1::BetaVector,
    b2::BetaVector;
    atol::Real=0.0,
    rtol::Real=Base.rtoldefault(b1.x, b1.y, atol),
    nans::Bool=false,
    norm::Function=abs,
)
    return isapprox(b1.x, b2.x; atol=atol, rtol=rtol, nans=nans, norm=norm) &&
           isapprox(b1.y, b2.y; atol=atol, rtol=rtol, nans=nans, norm=norm) &&
           isapprox(b1.z, b2.z; atol=atol, rtol=rtol, nans=nans, norm=norm)
end
