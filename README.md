# Math utilities for NEON, SSE and scalar implementation

Contains basic math routines for scalar implementations and NEON simd routines.
Everything implemented in single precision and double precision.
Almost all routines have *ULP* under 1.5 that is absolutely enough for media processing application (for some media
application it can be too high).
All methods reasonable fast for general purpose use. Performance comparable to libm, sometimes faster, sometimes slower,
but may be worse than CPU integrated solutions.
Have complementary NEON (double, double) type, and uint128.
Adds 64 bits integer arithmetics for SSE.

Implemented routines:

- [x] abs
- [x] acos
- [x] asin
- [x] atan
- [x] atan2
- [x] cbrt
- [x] floor
- [x] exp
- [ ] fmod
- [x] ln
- [x] hypot
- [x] pow
- [x] sin
- [x] cos
- [x] tan
- [x] sqrt
- [x] ceil

# Example

```rust
let value = 0.1f32.esin();

// For NEON simd
let value = vsinq_f32(vdupq_n_f32(0.1f32));
```

# Performance against libm

Sine Erydanos time:   [17.785 ns 17.884 ns 18.095 ns]                           
Sine libm time:   [27.928 ns 28.595 ns 29.398 ns]

Tan Erydanos time:   [27.593 ns 27.607 ns 27.621 ns]\
Tan libm time:   [28.854 ns 29.165 ns 29.467 ns]

Cbrt Erydanos time:   [23.260 ns 23.452 ns 23.650 ns]\
Cbrt Erydanos time:   [23.260 ns 23.452 ns 23.650 ns]

Pow Erydanos time:   [66.930 ns 67.465 ns 68.025 ns]\
Pow libm time:   [170.74 ns 172.71 ns 174.67 ns]

Asin Erydanos time:   [23.730 ns 23.953 ns 24.156 ns]\
Asin libm time:   [349.02 ns 350.39 ns 352.24 ns]

Atan Erydanos time:   [20.882 ns 21.115 ns 21.347 ns]\
Atan libm time:   [20.128 ns 20.309 ns 20.494 ns] 