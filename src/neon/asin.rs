/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::asin::{
    ASIN_POLY_10_D, ASIN_POLY_11_D, ASIN_POLY_12_D, ASIN_POLY_13_D, ASIN_POLY_14_D, ASIN_POLY_15_D,
    ASIN_POLY_16_D, ASIN_POLY_17_D, ASIN_POLY_18_D, ASIN_POLY_19_D, ASIN_POLY_1_D, ASIN_POLY_2_D,
    ASIN_POLY_3_D, ASIN_POLY_4_D, ASIN_POLY_5_D, ASIN_POLY_6_D, ASIN_POLY_7_D, ASIN_POLY_8_D,
    ASIN_POLY_9_D,
};
use crate::neon::general::{vcopysignq_f64, vmlafq_f64};
use std::arch::aarch64::*;

#[inline]
/// Computes asin with *ULP 1.5*
pub unsafe fn vasinq_f64(d: float64x2_t) -> float64x2_t {
    let ones = vdupq_n_f64(1f64);
    let v = vabsq_f64(d);
    let nan_mask = vcgtq_f64(v, ones);
    // for more 0.5
    let reverse_05_mask = vcgeq_f64(v, vdupq_n_f64(0.5f64));
    let reversed = vsqrtq_f64(vdivq_f64(vsubq_f64(ones, v), vdupq_n_f64(2f64)));
    let x = vbslq_f64(reverse_05_mask, reversed, v);
    let mut u = vdupq_n_f64(ASIN_POLY_19_D);
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_18_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_17_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_16_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_15_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_14_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_13_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_12_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_11_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_10_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_9_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_8_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_7_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_6_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_5_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_4_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_3_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_2_D));
    u = vmlafq_f64(u, x, vdupq_n_f64(ASIN_POLY_1_D));
    u = vmulq_f64(u, x);
    let j = u;
    let reconstruct_reversed = vmlafq_f64(
        vdupq_n_f64(-2f64),
        j,
        vdupq_n_f64(std::f64::consts::FRAC_PI_2),
    );
    let mut ret = vbslq_f64(reverse_05_mask, reconstruct_reversed, j);
    ret = vbslq_f64(nan_mask, vdupq_n_f64(f64::NAN), ret);
    vcopysignq_f64(ret, d)
}
