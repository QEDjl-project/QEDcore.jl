
#######
# General coordinate transformations
#######

"""
    AbstractCoordinateTransformation

Abstract base type for coordinate transformations supposed to be acting on four-momenta.
Every subtype of `trafo::AbstractCoordianteTransformation` should implement the following interface functions:

* `QEDcore._transform(trafo,p)`: transfroms `p`
* `Base.inv(trafo)`: returns the inverted transform

## Example

Implementing the interface by defining the interface functions:

```jldoctest trafo_interface
julia> using QEDcore

julia> struct TestTrafo{T} <: QEDcore.AbstractCoordinateTransformation
           a::T
       end

julia> QEDcore._transform(trafo::TestTrafo,p) = trafo.a*p

julia> Base.inv(trafo::TestTrafo) = TestTrafo(inv(trafo.a))

```

The `TestTrafo` can then be used to transform four-momenta:

```jldoctest trafo_interface
julia> trafo = TestTrafo(2.0)
TestTrafo{Float64}(2.0)

julia> p = SFourMomentum(4,3,2,1)
4-element SFourMomentum with indices SOneTo(4):
 4.0
 3.0
 2.0
 1.0

julia> trafo(p) # multiply every component with 2.0
4-element SFourMomentum with indices SOneTo(4):
 8.0
 6.0
 4.0
 2.0

julia> inv(trafo)(p) # divide every component by 2.0
4-element SFourMomentum with indices SOneTo(4):
 2.0
 1.5
 1.0
 0.5
```
"""
abstract type AbstractCoordinateTransformation end
Base.broadcastable(trafo::AbstractCoordinateTransformation) = Ref(trafo)

"""
    _transform(trafo::AbstractCoordinateTransformation,p::AbstractFourMomentum)

Interface function for the application of the transformation to the four-momentum `p`. Must return a four-momentum
of the same type as `p`.
"""
function _transform end

# make the transform callable
@inline function (trafo::AbstractCoordinateTransformation)(p::AbstractFourMomentum)
    return _transform(trafo, p)
end

@inline function (trafo::AbstractCoordinateTransformation)(
    psf::PSF
) where {PSF<:AbstractParticleStateful}
    p_prime = _transform(trafo, momentum(psf))
    return PSF(p_prime)
end

@inline function (trafo::AbstractCoordinateTransformation)(
    psp::PSP
) where {PSP<:AbstractPhaseSpacePoint}
    in_moms = momenta(psp, Incoming())
    out_moms = momenta(psp, Outgoing())
    in_moms_prime = _transform.(trafo, in_moms)
    out_moms_prime = _transform.(trafo, out_moms)

    proc = process(psp)
    mod = model(psp)
    ps_def = phase_space_definition(psp)
    return PhaseSpacePoint(proc, mod, ps_def, in_moms_prime, out_moms_prime)
end
