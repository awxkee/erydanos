/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::exp::*;
use crate::neon::general::{vmlafq_f64, vpow2ifq_s64};

#[inline]
pub unsafe fn vexpq_f64(d: float64x2_t) -> float64x2_t {
    let q = vcvtaq_s64_f64(vmulq_n_f64(d, R_LN2));
    let qf = vcvtq_f64_s64(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = vmlafq_f64(qf, vdupq_n_f64(-L2_U), d);
    r = vmlafq_f64(qf, vdupq_n_f64(-L2_L), r);
    let f = vmulq_f64(r, r);
    let mut u = vdupq_n_f64(EXP_POLY_10_D);
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_9_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_8_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_7_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_6_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_5_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_4_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_3_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_2_D));
    u = vmlafq_f64(u, f, vdupq_n_f64(EXP_POLY_1_D));
    let u = vaddq_f64(
        vdivq_f64(vmulq_n_f64(r, 2f64), vsubq_f64(u, r)),
        vdupq_n_f64(1f64),
    );
    let i2 = vreinterpretq_f64_s64(vpow2ifq_s64(q));
    let mut r = vmulq_f64(u, i2);
    r = vbslq_f64(vcltq_f64(d, vdupq_n_f64(-964f64)), vdupq_n_f64(0f64), r);
    r = vbslq_f64(
        vcgtq_f64(d, vdupq_n_f64(709f64)),
        vdupq_n_f64(f64::INFINITY),
        r,
    );
    r
}
