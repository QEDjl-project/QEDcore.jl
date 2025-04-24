using Random
using QEDbase
using QEDcore
using Test

RNG = MersenneTwister(137137)
N = 16

@testset "bisection with $FLOAT_T" for FLOAT_T in [Float32, Float64]
    TOL = eps(FLOAT_T)

    @testset "linear function $(a)*x - $(b)" for (a, b) in [rand(RNG, FLOAT_T, 2) for _ in 1:N]
        a += 4 * eps(FLOAT_T) # slope has to be at least slightly larger than 0
        test_f(x) = a * (x - b)    # root at b

        gt_root = b

        root1 = QEDcore._bisection(test_f, gt_root - 5, gt_root + 5; tol = TOL)
        root2 = QEDcore._bisection(test_f, gt_root + 5, gt_root - 5; tol = TOL)
        root3 = QEDcore._bisection(test_f, gt_root - 1000, gt_root + 1000; tol = TOL)
        root4 = QEDcore._bisection(test_f, gt_root + 1000, gt_root - 1000; tol = TOL)
        root5 = QEDcore._bisection(test_f, gt_root * (1 - eps(FLOAT_T)), gt_root * (1 + eps(FLOAT_T)); tol = TOL)

        for root in [root1, root2, root3, root4, root5]
            @test root isa FLOAT_T
            @test isapprox(test_f(root), 0; atol = TOL)
        end
    end

    @testset "quadratic functions $(a)*(x - $(x1))*(x - $(x2))" for (a, x1, x2) in [rand(RNG, FLOAT_T, 3) for _ in 1:N]
        a += 4 * eps(FLOAT_T)
        test_f(x) = a * (x - x1) * (x - x2) # roots x1 and x2

        @test_throws InvalidInputError QEDcore._bisection(test_f, FLOAT_T(-10), FLOAT_T(10); tol = TOL)    # f(a) * f(b) > 0 -> throw

        if (x2 < x1)    # swap so x1 < x2
            x1, x2 = x2, x1
        end

        gt_root_1 = x1

        root_x1_1 = QEDcore._bisection(test_f, gt_root_1 - 5, (gt_root_1 + x2) / 2; tol = TOL)
        root_x1_2 = QEDcore._bisection(test_f, (gt_root_1 + x2) / 2, gt_root_1 - 5; tol = TOL)
        root_x1_3 = QEDcore._bisection(test_f, gt_root_1 - 500, (gt_root_1 + x2) / 2; tol = TOL)
        root_x1_4 = QEDcore._bisection(test_f, (gt_root_1 + x2) / 2, gt_root_1 - 500; tol = TOL)
        root_x1_5 = QEDcore._bisection(test_f, gt_root_1 * (1 - eps(FLOAT_T)), gt_root_1 * (1 + eps(FLOAT_T)); tol = TOL)

        for root in [root_x1_1, root_x1_2, root_x1_3, root_x1_4, root_x1_5]
            @test root isa FLOAT_T
            @test isapprox(test_f(root), 0; atol = TOL)
        end

        gt_root_2 = x2

        root_x2_1 = QEDcore._bisection(test_f, gt_root_2 + 5, (x1 + x2) / 2; tol = TOL)
        root_x2_2 = QEDcore._bisection(test_f, (x1 + x2) / 2, gt_root_2 + 5; tol = TOL)
        root_x2_3 = QEDcore._bisection(test_f, gt_root_2 + 500, (x1 + x2) / 2; tol = TOL)
        root_x2_4 = QEDcore._bisection(test_f, (x1 + x2) / 2, gt_root_2 + 500; tol = TOL)
        root_x2_5 = QEDcore._bisection(test_f, gt_root_2 * (1 - eps(FLOAT_T)), gt_root_2 * (1 + eps(FLOAT_T)); tol = TOL)

        for root in [root_x2_1, root_x2_2, root_x2_3, root_x2_4, root_x2_5]
            @test root isa FLOAT_T
            @test isapprox(test_f(root), 0; atol = TOL)
        end
    end
end
