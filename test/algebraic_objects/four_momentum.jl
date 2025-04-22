using QEDcore
using Random

const ATOL = 1.0e-15

@testset "FourMomentum getter" for MomentumType in [SFourMomentum]
    rng = MersenneTwister(12345)
    x, y, z = rand(rng, 3)
    mass = rand(rng) + 0.5  # currently, very small masses break precisions
    E = hypot(x, y, z, mass)
    mom_onshell = MomentumType(E, x, y, z)
    mom_zero = MomentumType(0.0, 0.0, 0.0, 0.0)
    mom_offshell = MomentumType(0.0, 0.0, 0.0, mass)

    @testset "magnitude consistence" for mom in [mom_onshell, mom_offshell, mom_zero]
        @test getMagnitude2(mom) == getMag2(mom)
        @test getMagnitude(mom) == getMag(mom)
        @test isapprox(getMagnitude(mom), sqrt(getMagnitude2(mom)))
    end

    @testset "magnitude values" begin
        @test isapprox(getMagnitude2(mom_onshell), x^2 + y^2 + z^2)
        @test isapprox(getMagnitude(mom_onshell), hypot(x, y, z))
    end

    @testset "mass consistence" for mom_on in [mom_onshell, mom_zero]
        @test getInvariantMass2(mom_on) == getMass2(mom_on)
        @test getInvariantMass(mom_on) == getMass(mom_on)
        @test isapprox(getInvariantMass(mom_on), sqrt(getInvariantMass2(mom_on)))
    end

    @testset "mass value" begin
        @test isapprox(getInvariantMass2(mom_onshell), E^2 - (x^2 + y^2 + z^2))
        @test isapprox(getInvariantMass(mom_onshell), sqrt(E^2 - (x^2 + y^2 + z^2)))

        @test isapprox(getInvariantMass(mom_onshell), mass)
        @test isapprox(getInvariantMass(mom_offshell), -mass)
        @test isapprox(getInvariantMass(mom_zero), 0.0)
    end

    @testset "momentum components" begin
        @test getE(mom_onshell) == E
        @test getEnergy(mom_onshell) == getE(mom_onshell)
        @test getPx(mom_onshell) == x
        @test getPy(mom_onshell) == y
        @test getPz(mom_onshell) == z

        @test isapprox(getBeta(mom_onshell), hypot(x, y, z) / E)
        @test isapprox(getGamma(mom_onshell), 1 / sqrt(1.0 - getBeta(mom_onshell)^2))

        @test getE(mom_zero) == 0.0
        @test getEnergy(mom_zero) == 0.0
        @test getPx(mom_zero) == 0.0
        @test getPy(mom_zero) == 0.0
        @test getPz(mom_zero) == 0.0

        @test isapprox(getBeta(mom_zero), 0.0)
        @test isapprox(getGamma(mom_zero), 1.0)
    end

    @testset "transverse coordinates" for mom_on in [mom_onshell, mom_zero]
        @test getTransverseMomentum2(mom_on) == getPt2(mom_on)
        @test getTransverseMomentum2(mom_on) == getPerp2(mom_on)
        @test getTransverseMomentum(mom_on) == getPt(mom_on)
        @test getTransverseMomentum(mom_on) == getPerp(mom_on)

        @test isapprox(getPt(mom_on), sqrt(getPt2(mom_on)))

        @test getTransverseMass2(mom_on) == getMt2(mom_on)
        @test getTransverseMass(mom_on) == getMt(mom_on)
    end

    @testset "transverse coordinates value" begin
        @test isapprox(getTransverseMomentum2(mom_onshell), x^2 + y^2)
        @test isapprox(getTransverseMomentum(mom_onshell), hypot(x, y))
        @test isapprox(getTransverseMass2(mom_onshell), E^2 - z^2)
        @test isapprox(getTransverseMass(mom_onshell), sqrt(E^2 - z^2))
        @test isapprox(getMt(mom_offshell), -mass)
        @test isapprox(getRapidity(mom_onshell), 0.5 * log((E + z) / (E - z)))

        @test isapprox(getTransverseMomentum2(mom_zero), 0.0)
        @test isapprox(getTransverseMomentum(mom_zero), 0.0)
        @test isapprox(getTransverseMass2(mom_zero), 0.0)
        @test isapprox(getTransverseMass(mom_zero), 0.0)
        @test isapprox(getMt(mom_zero), 0.0)
    end

    @testset "spherical coordinates consistence" for mom_on in [mom_onshell, mom_zero]
        @test getRho2(mom_on) == getMagnitude2(mom_on)
        @test getRho(mom_on) == getMagnitude(mom_on)

        @test isapprox(getCosTheta(mom_on), cos(getTheta(mom_on)))
        @test isapprox(getCosPhi(mom_on), cos(getPhi(mom_on)))
        @test isapprox(getSinPhi(mom_on), sin(getPhi(mom_on)))
    end

    @testset "spherical coordinates values" begin
        @test isapprox(getTheta(mom_onshell), atan(getPt(mom_onshell), z))
        @test isapprox(getTheta(mom_zero), 0.0)

        @test isapprox(getPhi(mom_onshell), atan(y, x))
        @test isapprox(getPhi(mom_zero), 0.0)
    end

    @testset "light-cone coordinates" begin
        @test isapprox(getPlus(mom_onshell), 0.5 * (E + z))
        @test isapprox(getMinus(mom_onshell), 0.5 * (E - z))

        @test isapprox(getPlus(mom_zero), 0.0)
        @test isapprox(getMinus(mom_zero), 0.0)
    end
end # FourMomentum getter

const SCALE = 10.0 .^ [-9, 0, 5]
const M_MASSIVE = 1.0
const M_MASSLESS = 0.0

const M_ABSERR = 0.01
const M_RELERR = 0.0001

@testset "isonshell" begin
    rng = MersenneTwister(42)
    x_base, y_base, z_base = rand(rng, 3)

    @testset "correct onshell" begin
        @testset "($x_scale, $y_scale, $z_scale)" for (x_scale, y_scale, z_scale) in
            Iterators.product(SCALE, SCALE, SCALE)
            x, y, z = x_base * x_scale, y_base * y_scale, z_base * z_scale
            E_massless = hypot(x, y, z, M_MASSLESS)
            E_massive = hypot(x, y, z, M_MASSIVE)
            mom_massless = SFourMomentum(E_massless, x, y, z)
            mom_massive = SFourMomentum(E_massive, x, y, z)
            @test isonshell(mom_massless, M_MASSLESS)
            @test isonshell(mom_massive, M_MASSIVE)

            @test assert_onshell(mom_massless, M_MASSLESS) == nothing
            @test assert_onshell(mom_massive, M_MASSIVE) == nothing
        end
    end

    @testset "correct not onshell" begin
        @testset "$x_scale, $y_scale, $z_scale" for (x_scale, y_scale, z_scale) in
            Iterators.product(SCALE, SCALE, SCALE)
            x, y, z = x_base * x_scale, y_base * y_scale, z_base * z_scale
            m_err = min(M_ABSERR, M_RELERR * sum([x, y, z]) / 3.0) # mass error is M_RELERR of the mean of the components
            # but has at most the value M_ABSERR

            E_massless = hypot(x, y, z, (M_MASSLESS + m_err))
            E_massive = hypot(x, y, z, (M_MASSIVE + m_err))
            mom_massless = SFourMomentum(E_massless, x, y, z)
            mom_massive = SFourMomentum(E_massive, x, y, z)

            @test !isonshell(mom_massless, M_MASSLESS)
            @test !isonshell(mom_massive, M_MASSIVE)

            @test_throws OnshellError assert_onshell(mom_massless, M_MASSLESS)
            @test_throws OnshellError assert_onshell(mom_massive, M_MASSIVE)
        end
    end
end
