/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::ln::{
    LN_POLY_1_D, LN_POLY_2_D, LN_POLY_3_D, LN_POLY_4_D, LN_POLY_5_D, LN_POLY_6_D, LN_POLY_7_D,
    LN_POLY_8_D,
};
use crate::neon::general::{vilogb2kq_f64, visinfq_f64, visnanq_f64, vldexp3kq_f64, vmlafq_f64};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vlnq_f64(d: float64x2_t) -> float64x2_t {
    let n = vilogb2kq_f64(vmulq_n_f64(d, 1f64 / 0.75f64));
    let a = vldexp3kq_f64(d, vnegq_s64(n));
    let ones = vdupq_n_f64(1f64);
    let x = vdivq_f64(vsubq_f64(a, ones), vaddq_f64(a, ones));
    let x2 = vmulq_f64(x, x);
    let mut u = vdupq_n_f64(LN_POLY_8_D);
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_7_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_6_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_5_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_4_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_3_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_2_D));
    u = vmlafq_f64(u, x2, vdupq_n_f64(LN_POLY_1_D));
    let mut res = vmlafq_f64(
        vdupq_n_f64(std::f64::consts::LN_2),
        vcvtq_f64_s64(n),
        vmulq_f64(x, u),
    );
    // d == 0 -> -Inf
    res = vbslq_f64(vceqzq_f64(d), vdupq_n_f64(f64::NEG_INFINITY), res);
    // d == Inf -> Inf
    res = vbslq_f64(visinfq_f64(d), vdupq_n_f64(f64::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = vbslq_f64(
        vorrq_u64(vcltzq_f64(d), visnanq_f64(d)),
        vdupq_n_f64(f64::NAN),
        res,
    );
    res
}
