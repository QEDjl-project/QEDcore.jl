using QEDcore
using StaticArrays
using Random

include("../utils.jl")

RNG = MersenneTwister(708583836976)
ATOL = 1e-14
RTOL = 0.0
PHOTON_ENERGIES = (0.0, rand(RNG), rand(RNG) * 10)
COS_THETAS = (-1.0, -rand(RNG), 0.0, rand(RNG), 1.0)

# check every quadrant
PHIS = (
    0.0,
    rand(RNG) * pi / 2,
    pi / 2,
    (1.0 + rand(RNG)) * pi / 2,
    pi,
    (2 + rand(RNG)) * pi / 2,
    3 * pi / 2,
    (3 + rand(RNG)) * pi / 2,
    2 * pi,
)

X, Y, Z = rand(RNG, 3)

# test function to test scalar broadcasting
test_broadcast(x::AbstractParticle) = x
test_broadcast(x::ParticleDirection) = x
test_broadcast(x::AbstractSpinOrPolarization) = x

@testset "fermion" begin
    struct TestFermion <: Fermion end
    @test is_fermion(TestFermion())
    @test is_particle(TestFermion())
    @test !is_anti_particle(TestFermion())
    @test test_broadcast.(TestFermion()) == TestFermion()

    @testset "$p $d" for (p, d) in
                         Iterators.product((Electron, Positron), (Incoming, Outgoing))
        mom = SFourMomentum(sqrt(mass(p()) + X^2 + Y^2 + Z^2), X, Y, Z)
        particle_mass = mass(p())

        @testset "tooling" begin
            @test QEDbase._as_svec(base_state(p(), d(), mom, AllSpin())) isa SVector
            @test QEDbase._as_svec(base_state(p(), d(), mom, SpinUp())) isa SVector
            @test QEDbase._as_svec(base_state(p(), d(), mom, SpinDown())) isa SVector
        end
    end
    @testset "spinor properties" begin
        x, y, z = rand(RNG, 3)
        m = mass(Electron())
        P = SFourMomentum(sqrt(x^2 + y^2 + z^2 + m^2), x, y, z)

        U = base_state(Electron(), Incoming(), P, AllSpin())
        Ubar = base_state(Electron(), Outgoing(), P, AllSpin())
        V = base_state(Positron(), Outgoing(), P, AllSpin())
        Vbar = base_state(Positron(), Incoming(), P, AllSpin())

        @testset "normalization" begin
            for s1 in (1, 2)
                for s2 in (1, 2)
                    @test isapprox(Ubar[s1] * U[s2], 2 * m * (s1 == s2))
                    @test isapprox(Vbar[s1] * V[s2], -2 * m * (s1 == s2))
                    @test isapprox(Ubar[s1] * V[s2], 0.0)
                    @test isapprox(Vbar[s1] * U[s2], 0.0)
                end
            end
        end # normalization

        @testset "completeness" begin
            sumU = zero(DiracMatrix)
            sumV = zero(DiracMatrix)
            for spin in (1, 2)
                sumU += U[spin] * Ubar[spin]
                sumV += V[spin] * Vbar[spin]
            end

            @test isapprox(sumU, (slashed(P) + m * one(DiracMatrix)))
            @test isapprox(sumV, (slashed(P) - m * one(DiracMatrix)))
        end # completeness

        @testset "diracs equation" begin
            for spin in (1, 2)
                @test isapprox(
                    (slashed(P) - m * one(DiracMatrix)) * U[spin], zero(BiSpinor), atol=ATOL
                )
                @test isapprox(
                    (slashed(P) + m * one(DiracMatrix)) * V[spin], zero(BiSpinor), atol=ATOL
                )
                @test isapprox(
                    Ubar[spin] * (slashed(P) - m * one(DiracMatrix)),
                    zero(AdjointBiSpinor),
                    atol=ATOL,
                )
                @test isapprox(
                    Vbar[spin] * (slashed(P) + m * one(DiracMatrix)),
                    zero(AdjointBiSpinor),
                    atol=ATOL,
                )
            end
        end #diracs equation

        @testset "sandwich" begin
            for s1 in (1, 2)
                for s2 in (1, 2)
                    @test isapprox(Ubar[s1] * (gamma() * U[s2]), 2 * P * (s1 == s2))
                end
            end
        end #sandwich
    end
end

@testset "photon" begin
    @test !is_fermion(Photon())
    @test is_boson(Photon())
    @test is_particle(Photon())
    @test is_anti_particle(Photon())
    @test charge(Photon()) == 0.0
    @test mass(Photon()) == 0.0
    @test test_broadcast.(Photon()) == Photon()

    @testset "$D" for D in [Incoming, Outgoing]
        @testset "$om $cth $phi" for (om, cth, phi) in
                                     Iterators.product(PHOTON_ENERGIES, COS_THETAS, PHIS)
            #@testset "$x $y $z" for (x,y,z) in Iterators.product(X_arr,Y_arr,Z_arr)

            mom = SFourMomentum(_cartesian_coordinates(om, om, cth, phi))
            both_photon_states = base_state(Photon(), D(), mom, AllPolarization())

            # property test the photon states
            @test isapprox((both_photon_states[1] * mom), 0.0, atol=ATOL, rtol=RTOL)
            @test isapprox((both_photon_states[2] * mom), 0.0, atol=ATOL, rtol=RTOL)
            @test isapprox(
                (both_photon_states[1] * both_photon_states[1]), -1.0, atol=ATOL, rtol=RTOL
            )
            @test isapprox(
                (both_photon_states[2] * both_photon_states[2]), -1.0, atol=ATOL, rtol=RTOL
            )
            @test isapprox(
                (both_photon_states[1] * both_photon_states[2]), 0.0, atol=ATOL, rtol=RTOL
            )

            # test the single polarization states
            @test base_state(Photon(), D(), mom, PolarizationX()) == both_photon_states[1]
            @test base_state(Photon(), D(), mom, PolarizationY()) == both_photon_states[2]
            @test base_state(Photon(), D(), mom, PolX()) == both_photon_states[1]
            @test base_state(Photon(), D(), mom, PolY()) == both_photon_states[2]

            @test QEDbase._as_svec(base_state(Photon(), D(), mom, PolX())) isa SVector
            @test QEDbase._as_svec(base_state(Photon(), D(), mom, PolY())) isa SVector
            @test QEDbase._as_svec(base_state(Photon(), D(), mom, AllPol())) isa SVector

            @test QEDbase._as_svec(base_state(Photon(), D(), mom, PolX()))[1] ==
                both_photon_states[1]
            @test QEDbase._as_svec(base_state(Photon(), D(), mom, PolY()))[1] ==
                both_photon_states[2]
            @test QEDbase._as_svec(base_state(Photon(), D(), mom, AllPol()))[1] ==
                both_photon_states[1]
            @test QEDbase._as_svec(base_state(Photon(), D(), mom, AllPol()))[2] ==
                both_photon_states[2]
        end
    end
end
