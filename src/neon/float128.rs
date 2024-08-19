/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::neon::general::{vmlafq_f64, vmlsfq_f64};

/// Type represents f128, in low f64 and high f64 part
/// We'll keep this splat because mainly it is required to perform operations in both of them sequentially
/// This is not real f128, this should be considered as `double double` rather than IEEE binary128
///
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct float128x2_t(float64x2_t, float64x2_t);

#[inline]
unsafe fn vupperpartq_f64(a: float64x2_t) -> float64x2_t {
    let mask = vdupq_n_u64(0x_ffff_ffff_f800_0000);
    vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a), mask))
}

#[inline]
/// Performs multiplication for f128
pub unsafe fn vmulq_f128(a: float128x2_t, b: float128x2_t) -> float128x2_t {
    let xh = vupperpartq_f64(a.0);
    let xl = vsubq_f64(a.0, xh);
    let yh = vupperpartq_f64(b.0);
    let yl = vsubq_f64(b.0, yh);
    let r0 = vmulq_f64(a.0, b.0);
    let v0 = vmlsfq_f64(xl, yl, r0);
    let j0 = vmlafq_f64(xh, yh, vmlafq_f64(xl, yh, vmlafq_f64(xh, yl, v0)));
    let j2 = vmlafq_f64(a.0, b.1, vmlafq_f64(a.1, b.0, j0));
    float128x2_t(r0, j2)
}

#[inline]
/// Widens f64 into f128 and performs multiplication
pub unsafe fn vmullq_f64(a: float64x2_t, b: float64x2_t) -> float128x2_t {
    let xh = vupperpartq_f64(a);
    let xl = vsubq_f64(a, xh);
    let yh = vupperpartq_f64(b);
    let yl = vsubq_f64(b, yh);
    let r0 = vmulq_f64(a, b);
    let d0 = r0;
    let j0 = vmlafq_f64(
        xh,
        yh,
        vmlafq_f64(xl, yh, vmlafq_f64(xh, yl, vmulq_f64(xl, yl))),
    );
    let d1 = vsubq_f64(j0, r0);
    float128x2_t(d0, d1)
}

#[inline]
/// Adds f64 with widening to f128
pub unsafe fn vaddl_f64(a: float64x2_t, b: float64x2_t) -> float128x2_t {
    let r0 = vaddq_f64(a, b);
    let v = vsubq_f64(r0, b);
    float128x2_t(
        r0,
        vaddq_f64(vsubq_f64(a, vsubq_f64(r0, v)), vsubq_f64(b, v)),
    )
}

/// Adds f128 to another f128
#[inline]
pub unsafe fn vaddq_f128(a: float128x2_t, b: float128x2_t) -> float128x2_t {
    let r0 = vaddq_f64(a.0, b.0);
    float128x2_t(
        r0,
        vsubq_f64(vaddq_f64(vaddq_f64(vaddq_f64(b.0, a.1), r0), b.1), r0),
    )
}

/// Negates f128
#[inline]
pub unsafe fn vnegq_f128(a: float128x2_t) -> float128x2_t {
    float128x2_t(vnegq_f64(a.0), vnegq_f64(a.1))
}

/// Adds f64 with widening to f128
#[inline]
pub unsafe fn vaddw_f64(a: float128x2_t, b: float64x2_t) -> float128x2_t {
    let r0 = vaddq_f64(a.0, b);
    let v = vsubq_f64(r0, a.0);
    float128x2_t(
        r0,
        vaddq_f64(
            vaddq_f64(vsubq_f64(a.0, vsubq_f64(r0, v)), vsubq_f64(b, v)),
            a.1,
        ),
    )
}

#[inline]
pub unsafe fn vdivq_f128(a: float128x2_t, b: float128x2_t) -> float128x2_t {
    let t = vdivq_f64(vdupq_n_f64(1.), b.0);
    let dh = vupperpartq_f64(b.0);
    let dl = vsubq_f64(b.0, dh);
    let th = vupperpartq_f64(t);
    let tl = vsubq_f64(t, th);
    let nhh = vupperpartq_f64(a.0);
    let nhl = vsubq_f64(a.0, nhh);

    let q0 = vmulq_f64(a.0, t);

    let j0 = vmlafq_f64(
        nhh,
        th,
        vmlafq_f64(nhh, tl, vmlafq_f64(nhl, th, vmulq_f64(nhl, tl))),
    );
    let j1 = vmlafq_f64(
        dh,
        th,
        vmlafq_f64(dh, tl, vmlafq_f64(dl, th, vmulq_f64(dl, tl))),
    );
    let j2 = vsubq_f64(vdupq_n_f64(1f64), j1);
    let u = vsubq_f64(vmlafq_f64(q0, j2, j0), q0);
    let low = vmlafq_f64(vsubq_f64(a.1, vmulq_f64(q0, b.1)), t, u);
    float128x2_t(q0, low)
}

#[inline]
/// Converts f128 into f64
pub unsafe fn vcvtq_f64_f128(d: float128x2_t) -> float64x2_t {
    let j0 = vaddq_f64(d.0, d.1);
    j0
}

#[inline]
/// Converts f64 into f128
pub unsafe fn vcvtq_f128_f64(d: float64x2_t) -> float128x2_t {
    float128x2_t(d, vdupq_n_f64(0f64))
}

#[inline]
/// Converts f64 into f128
pub unsafe fn vdupq_n_f128(d: f64) -> float128x2_t {
    float128x2_t(vdupq_n_f64(d), vdupq_n_f64(0f64))
}

/// Computes f128 as f64 and extracts in general register
#[inline]
pub unsafe fn vextractq_f128<const LANE: i32>(d: float128x2_t) -> f64 {
    let product = vaddq_f64(d.0, d.1);
    vgetq_lane_f64::<LANE>(product)
}

#[inline]
/// Fused multiply add for f128
pub unsafe fn vmlafq_f128(a: float128x2_t, b: float128x2_t, c: float128x2_t) -> float128x2_t {
    vaddq_f128(vmulq_f128(a, b), c)
}
