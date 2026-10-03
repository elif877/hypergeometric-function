"""Tests for hypergeometric_2f1.

Conventions in this file:
* We compare floats with math.isclose using rel_tol and abs_tol rather than ==.
* Reference values come from identities that the implementation is *expected*
  to satisfy (geometric series, binomial theorem, quadratic transformation),
  not from an external table we cannot verify offline.
"""

import math
import unittest

from hypergeometric_function import hypergeometric_2f1


class TestTrivialCases(unittest.TestCase):

    def test_value_at_zero_is_one(self):
        # 2F1(a,b;c;0) = 1 identically.
        self.assertAlmostEqual(hypergeometric_2f1(1.0, 1.0, 1.0, 0.0), 1.0)
        self.assertAlmostEqual(hypergeometric_2f1(-3.0, 2.5, 1.5, 0.0), 1.0)
        self.assertAlmostEqual(hypergeometric_2f1(0.0, 0.0, 5.0, 0.0), 1.0)

    def test_zero_parameters(self):
        # 2F1(0,b;c;z) = 1 because (0)_n = 0 for n >= 1.
        self.assertAlmostEqual(hypergeometric_2f1(0.0, 3.0, 2.0, 0.7), 1.0)
        self.assertAlmostEqual(hypergeometric_2f1(3.0, 0.0, 2.0, 0.7), 1.0)


class TestKnownIdentities(unittest.TestCase):

    def test_geometric_series(self):
        # 2F1(1,1;1;z) = 1/(1-z) for |z| < 1.
        for z in (-0.5, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            with self.subTest(z=z):
                got = hypergeometric_2f1(1.0, 1.0, 1.0, z)
                expected = 1.0 / (1.0 - z)
                self.assertTrue(math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15),
                                msg=f"z={z}: got {got}, expected {expected}")

    def test_binomial_theorem(self):
        # 2F1(-n, b; b; z) = (1-z)^n for non-negative integer n and b != 0.
        cases = [
            (-1, 2.0, 2.0),
            (-2, 0.5, 0.5),
            (-3, 1.7, 1.7),
            (-5, 3.3, 3.3),
        ]
        for n, b, c in cases:
            for z in (-0.4, 0.0, 0.25, 0.5, 0.9):
                with self.subTest(n=n, b=b, z=z):
                    got = hypergeometric_2f1(float(n), b, c, z)
                    expected = (1.0 - z) ** (-n)
                    self.assertTrue(
                        math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15),
                        msg=f"n={n}, z={z}: got {got}, expected {expected}",
                    )

    def test_logarithmic_case(self):
        # 2F1(1,1;2;z) = -ln(1-z)/z for z != 0; equals 1 at z=0.
        for z in (-0.5, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            with self.subTest(z=z):
                got = hypergeometric_2f1(1.0, 1.0, 2.0, z)
                expected = -math.log(1.0 - z) / z
                self.assertTrue(math.isclose(got, expected, rel_tol=1e-11, abs_tol=1e-13),
                                msg=f"z={z}: got {got}, expected {expected}")

    def test_pfaff_transformation_identity(self):
        # Pfaff: 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1)).
        # Check the function honours this by comparing the two sides via the
        # public API at a point where both series converge directly.
        a, b, c = 0.7, 1.3, 1.9
        z = 0.4
        lhs = hypergeometric_2f1(a, b, c, z)
        # RHS evaluated explicitly: transform and call again.
        w = z / (z - 1.0)
        rhs = (1.0 - z) ** (-a) * hypergeometric_2f1(a, c - b, c, w)
        self.assertTrue(math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-14),
                        msg=f"Pfaff identity failed: lhs={lhs}, rhs={rhs}")


class TestNegativeArgument(unittest.TestCase):

    def test_negative_z_via_pfaff(self):
        # For z < 0 the Pfaff point w = z/(z-1) lies in (0, 1) and is smaller
        # in modulus than z when z < -1.  The implementation should still
        # produce the correct value; use the geometric series as the oracle.
        # 2F1(1,1;1;z) = 1/(1-z) for all z < 1.
        for z in (-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -50.0):
            with self.subTest(z=z):
                got = hypergeometric_2f1(1.0, 1.0, 1.0, z)
                expected = 1.0 / (1.0 - z)
                self.assertTrue(math.isclose(got, expected, rel_tol=1e-11, abs_tol=1e-13),
                                msg=f"z={z}: got {got}, expected {expected}")


class TestErrorConditions(unittest.TestCase):

    def test_c_non_positive_integer_raises(self):
        for c in (0.0, -1.0, -2.0, -5.0, -10.0):
            with self.subTest(c=c):
                with self.assertRaises(ValueError):
                    hypergeometric_2f1(1.0, 1.0, c, 0.3)

    def test_z_on_branch_cut_raises(self):
        # z >= 1 is on or past the branch cut [1, +infty).
        for z in (1.0, 1.5, 2.0, 10.0, 100.0):
            with self.subTest(z=z):
                with self.assertRaises(ValueError):
                    hypergeometric_2f1(1.0, 1.0, 1.0, z)

    def test_non_finite_z_raises(self):
        for z in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(z=z):
                with self.assertRaises(ValueError):
                    hypergeometric_2f1(1.0, 1.0, 1.0, z)


class TestPolynomialTermination(unittest.TestCase):

    def test_first_order_polynomial(self):
        # 2F1(-1, b; c; z) = 1 - (b/c) z.
        got = hypergeometric_2f1(-1.0, 2.0, 3.0, 0.6)
        expected = 1.0 - (2.0 / 3.0) * 0.6
        self.assertTrue(math.isclose(got, expected, rel_tol=1e-14, abs_tol=1e-16))

    def test_second_order_polynomial(self):
        # 2F1(-2, b; c; z) = 1 - 2(b/c)z + (b(b+1))/(c(c+1)) z^2 / 1! ... wait,
        # explicit: sum_{n=0}^{2} (-2)_n (b)_n / (c)_n * z^n / n!
        a, b, c, z = -2.0, 1.5, 2.5, 0.4
        t0 = 1.0
        t1 = (-2.0) * b / c * z
        t2 = ((-2.0) * (-1.0)) * (b * (b + 1.0)) / (c * (c + 1.0)) * (z ** 2) / 2.0
        expected = t0 + t1 + t2
        got = hypergeometric_2f1(a, b, c, z)
        self.assertTrue(math.isclose(got, expected, rel_tol=1e-14, abs_tol=1e-16))


class TestNumericalStability(unittest.TestCase):

    def test_near_boundary_converges(self):
        # z close to 1 from below; series converges slowly but should still
        # be close to the geometric-series oracle.
        z = 0.999
        got = hypergeometric_2f1(1.0, 1.0, 1.0, z)
        expected = 1.0 / (1.0 - z)
        self.assertTrue(math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9))

    def test_large_negative_with_transformation(self):
        # z = -100; direct series diverges, Pfaff must be used.
        # 2F1(1,1;1;z) = 1/(1-z) = 1/101.
        got = hypergeometric_2f1(1.0, 1.0, 1.0, -100.0)
        expected = 1.0 / 101.0
        self.assertTrue(math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
