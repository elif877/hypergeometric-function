# hypergeometric_function

Evaluates the Gaussian hypergeometric function 2F1(a, b; c; z) for real
parameters and real argument, using the defining power series inside the unit
disk and a single Pfaff transformation for analytic continuation outside it.

## Usage

```python
from hypergeometric_function import hypergeometric_2f1

# Geometric series: 2F1(1,1;1;z) = 1/(1-z)
print(hypergeometric_2f1(1.0, 1.0, 1.0, 0.5))   # 2.0

# Logarithmic case: 2F1(1,1;2;z) = -ln(1-z)/z
print(hypergeometric_2f1(1.0, 1.0, 2.0, 0.5))  # ~1.3862944

# Negative argument: handled via the Pfaff map z -> z/(z-1)
print(hypergeometric_2f1(1.0, 1.0, 1.0, -10.0))  # 1/11
```

## Why this exists

The hypergeometric function appears everywhere — binomial coefficients,
incomplete beta, elliptic integrals, solutions of second-order ODEs — and the
standard libraries that evaluate it (SciPy, mpmath) are heavy dependencies.
This package is a zero-dependency, standard-library-only implementation sized
for cases where you need a handful of 2F1 evaluations and do not want to ship a
full numerical backend.

The trade-off: instead of implementing the full analytic-continuation ladder
( linear + quadratic + Pfaff + connection formulas ), this library uses exactly
one continuation step — the Pfaff transformation
`2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))`.  For real `z < 1` that is
sufficient: the transformed argument always lands inside the unit disk, so the
series converges.  Complex arguments are not supported, and the branch cut
`[1, +infty)` is rejected rather than silently picking a side.

## Edge cases you will hit

* `c` must not be a non-positive integer (`0, -1, -2, ...`).  The Pochhammer
  denominator `(c)_n` vanishes there and the function is undefined; the library
  raises `ValueError`.
* `z >= 1` raises `ValueError`.  The principal branch of `(1-z)^{-a}` has a cut
  on `[1, +infty)`; rather than return a one-sided limit without being asked,
  the library refuses.  If you need a value on the cut, approach it from above
  or below explicitly (e.g. evaluate at `z - 1e-9` and `z + 1e-9 - 1e-9j` once
  complex support is added).
* Very near `z = 1` from below, the series converges slowly.  The library caps
  iteration count and uses a relative tolerance, so results within `1e-9` are
  expected but more precision than that is not guaranteed at `|1 - z| < 1e-3`.

## Exported names

* `hypergeometric_2f1(a: float, b: float, c: float, z: float) -> float`

No other names are part of the public API.
