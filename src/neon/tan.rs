/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::neon::general::vmlafq_f64;
use crate::sin::{PI_A2, PI_B2};
use crate::tan::{
    TAN_POLY_1_D, TAN_POLY_2_D, TAN_POLY_3_D, TAN_POLY_4_D, TAN_POLY_5_D, TAN_POLY_6_D,
    TAN_POLY_7_D, TAN_POLY_8_D, TAN_POLY_9_D,
};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
/// Computes tan with error bound *ULP 2.0*
pub unsafe fn vtanq_f64(d: float64x2_t) -> float64x2_t {
    let q = vcvtaq_s64_f64(vmulq_n_f64(d, std::f64::consts::FRAC_2_PI));
    let qf = vcvtq_f64_s64(q);

    let mut x = vmlafq_f64(qf, vdupq_n_f64(-PI_A2 * 0.5), d);
    x = vmlafq_f64(qf, vdupq_n_f64(-PI_B2 * 0.5), x);

    let even = vceqzq_s64(vandq_s64(q, vdupq_n_s64(1)));
    x = vbslq_f64(even, x, vnegq_f64(x));
    x = vmulq_n_f64(x, 0.5f64);
    let x2 = vmulq_f64(x, x);
    let mut u = vdupq_n_f64(TAN_POLY_9_D);
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_8_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_7_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_6_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_5_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_4_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_3_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_2_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(TAN_POLY_1_D));
    u = vmlafq_f64(u, vmulq_f64(x2, x), x);
    u = vdivq_f64(
        vmulq_n_f64(u, 2.),
        vsubq_f64(vdupq_n_f64(1.), vmulq_f64(u, u)),
    );
    u = vbslq_f64(even, u, vdivq_f64(vdupq_n_f64(1.), u));
    u
}
