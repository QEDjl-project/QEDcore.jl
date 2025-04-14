# dummy particles
struct TestParticle <: AbstractParticleType end # generic particle
struct TestParticleFermion <: FermionLike end
QEDbase.mass(::Type{T}, ::TestParticleFermion) where {T <: Number} = T(1.2)
struct TestParticleBoson <: BosonLike end
QEDbase.mass(::Type{T}, ::TestParticleBoson) where {T <: Number} = T(2.3)

const PARTICLE_SET = [TestParticleFermion(), TestParticleBoson()]
