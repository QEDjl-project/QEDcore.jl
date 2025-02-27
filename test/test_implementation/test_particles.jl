
# dummy particles
struct TestParticle <: AbstractParticleType end # generic particle
struct TestParticleFermion <: FermionLike end
QEDbase.mass(::TestParticleFermion) = 1.2
struct TestParticleBoson <: BosonLike end
QEDbase.mass(::TestParticleBoson) = 2.3

const PARTICLE_SET = [TestParticleFermion(), TestParticleBoson()]
