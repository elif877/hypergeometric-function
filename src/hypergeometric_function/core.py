"""Gaussian hypergeometric function 2F1(a, b; c; z).

This module evaluates the Gaussian (Gauss) hypergeometric function

    F(a, b; c; z) = sum_{n=0}^{infty} [(a)_n (b)_n / (c)_n] z^n / n!

where (q)_n is the Pochhammer symbol (rising factorial).

Convergence and analytic-continuation strategy
---------------------------------------------
The defining power series converges absolutely for |z| < 1.  The standard
Euler/Pfaff transformations

    2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))
    2F1(a,b;c;z) = (1-z)^{-b} 2F1(c-a, b; c; z/(z-1))

map the z-plane so that the *argument of the resulting 2F1* lands inside the
unit disk for a much larger region of the original z.  Specifically, after the
Pfaff map w = z/(z-1) we have |w| < 1 whenever Re(z) < 1/2.  We pick whichever
of z and z/(z-1) has smaller modulus and apply the corresponding
transformation, then sum the series at the reduced point.  This single
continuation step handles every z with |arg(1-z)| < pi (i.e. everything except
the branch cut [1, +infty)).

Choices made explicitly
-----------------------
* Parameters a, b, c, z may be int or float; complex inputs are not supported
  (the brief asks for convergent series + one continuation step, and complex
  arithmetic would enlarge the surface area without changing the algorithm).
* c must not be a non-positive integer (0, -1, -2, ...); (c)_n would vanish and
  the series is undefined.  We raise ValueError.
* The branch cut [1, +infty) on the real axis is rejected: the principal branch
  of (1-z)^{-a} is discontinuous there, so no single analytic value is correct
  without a side specification.  We raise ValueError and ask the caller to
  approach from above or below explicitly.
* Polynomial cases where a or b is a non-positive integer terminate the series;
  we detect this and stop, returning an exact rational-ish float.
* Convergence is by term-magnitude threshold; we also cap iterations to avoid
  an infinite loop on a slowly-convergent or non-convergent argument.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def _pochhammer_next(prev_term: float, n: int, a: float, b: float, c: float, z: float) -> float:
    """Compute term n+1 from term n using the multiplicative recurrence.

    t_{n+1} / t_n = (a+n)(b+n) / ((c+n)(n+1)) * z

    This avoids recomputing factorials / Pochhammer symbols each step and keeps
    the running product numerically stable for moderate n.
    """
    return prev_term * (a + n) * (b + n) * z / ((c + n) * (n + 1))


def _series_sum(a: float, b: float, c: float, z: float,
                max_terms: int = 100000, tol: float = 1e-15) -> Tuple[float, int]:
    """Sum the 2F1 series at |z| < 1 (or at a point mapped there).

    Returns (value, terms_used).  Raises ValueError if c is a non-positive
    integer, since (c)_n would be zero for some n.
    """
    # c in {0, -1, -2, ...} => singular denominator.
    if c <= 0 and abs(c - round(c)) < 1e-14 and float(round(c)) == c:
        raise ValueError(f"parameter c must not be a non-positive integer, got c={c!r}")

    total = 1.0          # n = 0 term is 1.
    term = 1.0
    n = 0
    while n < max_terms:
        new_term = _pochhammer_next(term, n, a, b, c, z)
        total += new_term
        # Stop when the increment is negligible relative to the running total,
        # OR when it is absolutely tiny (covers total ~= 0).
        if abs(new_term) <= tol * max(abs(total), 1.0):
            # One more step in case the next term would have been the final
            # significant contribution; standard practice for these sums.
            n += 1
            term = new_term
            break
        term = new_term
        n += 1
    return total, n


def hypergeometric_2f1(a: float, b: float, c: float, z: float) -> float:
    """Evaluate the Gaussian hypergeometric function 2F1(a, b; c; z).

    Parameters
    ----------
    a, b, c : float
        Parameters of the function.  ``c`` must not be a non-positive integer
        (0, -1, -2, ...); doing so raises ``ValueError`` because the Pochhammer
        denominator (c)_n vanishes.
    z : float
        Argument.  Must satisfy ``z < 1`` on the real line; the principal branch
        has a branch cut on [1, +infty) and this implementation refuses to pick
        a side silently, raising ``ValueError`` for ``z >= 1``.

    Returns
    -------
    float
        The value 2F1(a, b; c; z) on the principal branch.

    Notes
    -----
    For |z| < 1 the defining series is summed directly.  For z outside the unit
    disk but below the branch cut (i.e. z < 1, including z < 0), the Pfaff
    transformation w = z/(z-1) is applied; |w| < 1 in this region, so the series
    converges rapidly at the transformed argument.
    """
    # ---- Validate c -------------------------------------------------------
    if c == 0.0 or (c < 0 and float(round(c)) == c and abs(c - round(c)) < 1e-14):
        raise ValueError(f"c must not be a non-positive integer; got c={c!r}")

    # ---- Validate z against the branch cut -------------------------------
    # The principal branch of (1-z)^{-a} has a cut on [1, +infty).  Refusing
    # z >= 1 is the honest choice: returning a one-sided limit without being
    # asked for one would mislead callers who expect analyticity.
    if not math.isfinite(z) or z >= 1.0:
        raise ValueError(
            f"z must be finite and < 1 (principal branch); got z={z!r}"
        )

    # ---- Trivial exact cases ---------------------------------------------
    # 2F1(a, b; c; 0) = 1 for all valid a, b, c.
    if z == 0.0:
        return 1.0

    # ---- Choose evaluation point via Pfaff if it helps --------------------
    # Pfaff: 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))
    # For real z < 1, |z/(z-1)| < 1 exactly when z < 1/2; but the *ratio*
    # |z/(z-1)| < |z| also holds for z in (-infty, 0).  We pick whichever of
    # z and w = z/(z-1) has smaller absolute value.
    w = z / (z - 1.0) if z != 1.0 else float("inf")

    if abs(w) < abs(z):
        # Evaluate at w via Pfaff with the 'a' parameter pulled out.
        prefactor = (1.0 - z) ** (-a)
        inner, _ = _series_sum(a, c - b, c, w)
        return prefactor * inner

    # Direct series at z.
    value, _ = _series_sum(a, b, c, z)
    return value
