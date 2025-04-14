###############
# The particle types
#
# In this file, we define the types of particles, their states and directions, and
# implement the abstact particle interface accordingly.
###############

"""
Abstract base types for particle species that act like fermions in the sense of particle statistics.

!!! note "particle interface"
    Every concrete subtype of [`FermionLike`](@ref) has `is_fermion(::FermionLike) = true`.
"""
abstract type FermionLike <: AbstractParticleType end

QEDbase.is_fermion(::FermionLike) = true

"""
Abstract base type for fermions as distinct from [`AntiFermion`](@ref)s.

!!! note "particle interface"
    All subtypes of `Fermion` have
    ```julia
    is_fermion(::Fermion) = true
    is_particle(::Fermion) = true
    is_anti_particle(::Fermion) = false
    ```

"""
abstract type Fermion <: FermionLike end

QEDbase.is_particle(::Fermion) = true

QEDbase.is_anti_particle(::Fermion) = false

"""
Abstract base type for anti-fermions as distinct from its particle counterpart `Fermion`.

!!! note "particle interface"
    All subtypes of `AntiFermion` have
    ```julia
    is_fermion(::AntiFermion) = true
    is_particle(::AntiFermion) = false
    is_anti_particle(::AntiFermion) = true
    ```
"""
abstract type AntiFermion <: FermionLike end

QEDbase.is_particle(::AntiFermion) = false

QEDbase.is_anti_particle(::AntiFermion) = true

"""
Abstract base type for majorana-fermions, i.e. fermions which are their own anti-particles.

!!! note "particle interface"
    All subtypes of `MajoranaFermion` have
    ```julia
    is_fermion(::MajoranaFermion) = true
    is_particle(::MajoranaFermion) = true
    is_anti_particle(::MajoranaFermion) = true
    ```

"""
abstract type MajoranaFermion <: FermionLike end

QEDbase.is_particle(::MajoranaFermion) = true

QEDbase.is_anti_particle(::MajoranaFermion) = true

"""
Concrete type for *electrons* as a particle species. Mostly used for dispatch.

```jldoctest
julia> using QEDcore

julia> Electron()
electron
```

!!! note "particle interface"
    Besides being a subtype of [`Fermion`](@ref), objects of type `Electron` have

    ```julia
    mass(::Type{T}, ::Electron) = one(T)
    charge(::Type{T}, ::Electron) = -one(T)
    ```
"""
struct Electron <: Fermion end
QEDbase.mass(::Type{T}, ::Electron) where {T} = one(T)
QEDbase.charge(::Type{T}, ::Electron) where {T} = -one(T)
Base.show(io::IO, ::Electron) = print(io, "electron")

"""
Concrete type for *positrons* as a particle species. Mostly used for dispatch.

```jldoctest
julia> using QEDcore

julia> Positron()
positron
```

!!! note "particle interface"
    Besides being a subtype of [`AntiFermion`](@ref), objects of type `Positron` have

    ```julia
    mass(::Type{T}, ::Positron) = one(T)
    charge(::Type{T}, ::Positron) = one(T)
    ```
"""
struct Positron <: AntiFermion end
QEDbase.mass(::Type{T}, ::Positron) where {T} = one(T)
QEDbase.charge(::Type{T}, ::Positron) where {T} = one(T)
Base.show(io::IO, ::Positron) = print(io, "positron")

"""
Abstract base types for particle species that act like bosons in the sense of particle statistics.

!!! note "particle interface"
    Every concrete subtype of `BosonLike` has `is_boson(::BosonLike) = true`.
"""
abstract type BosonLike <: AbstractParticleType end

QEDbase.is_boson(::BosonLike) = true

"""
Abstract base type for bosons as distinct from its anti-particle counterpart [`AntiBoson`](@ref).

!!! note "particle interface"
    All subtypes of `Boson` have
    ```julia
    is_boson(::Boson) = true
    is_particle(::Boson) = true
    is_anti_particle(::Boson) = false
    ```
"""
abstract type Boson <: BosonLike end
QEDbase.is_particle(::Boson) = true
QEDbase.is_anti_particle(::Boson) = false

"""
Abstract base type for anti-bosons as distinct from its particle counterpart [`Boson`](@ref).

!!! note "particle interface"
    All subtypes of `AntiBoson` have
    ```julia
    is_boson(::AntiBoson) = true
    is_particle(::AntiBoson) = false
    is_anti_particle(::AntiBoson) = true
    ```
"""
abstract type AntiBoson <: BosonLike end
QEDbase.is_particle(::AntiBoson) = false
QEDbase.is_anti_particle(::AntiBoson) = true

"""
Abstract base type for majorana-bosons, i.e. bosons which are their own anti-particles.

!!! note "particle interface"
    All subtypes of `MajoranaBoson` have
    ```julia
    is_boson(::MajoranaBoson) = true
    is_particle(::MajoranaBoson) = true
    is_anti_particle(::MajoranaBoson) = true
    ```
"""
abstract type MajoranaBoson <: BosonLike end
QEDbase.is_particle(::MajoranaBoson) = true
QEDbase.is_anti_particle(::MajoranaBoson) = true

"""
Concrete type for the *photons* as a particle species. Mostly used for dispatch.

```jldoctest
julia> using QEDcore

julia> Photon()
photon
```

!!! note "particle interface"
    Besides being a subtype of `MajoranaBoson`, `Photon` has

    ```julia
    mass(::Type{T}, ::Photon) = zero(T)
    charge(::Type{T}, ::Photon) = zero(T)
    ```
"""
struct Photon <: MajoranaBoson end
QEDbase.mass(::Type{T}, ::Photon) where {T} = zero(T)
QEDbase.charge(::Type{T}, ::Photon) where {T} = zero(T)
Base.show(io::IO, ::Photon) = print(io, "photon")
