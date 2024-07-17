/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{visinfq_f64, visnanq_f64, vmlafq_f64};
use std::arch::aarch64::*;

/// Computes 2D Euclidian Distance *ULP 0.5*
#[inline]
pub unsafe fn vhypotq_f64(x: float64x2_t, y: float64x2_t) -> float64x2_t {
    let x = vabsq_f64(x);
    let y = vabsq_f64(y);
    let max = vmaxq_f64(x, y);
    let min = vminq_f64(x, y);
    let r = vdivq_f64(min, max);
    let mut ret = vmulq_f64(vsqrtq_f64(vmlafq_f64(r, r, vdupq_n_f64(1f64))), max);
    let is_any_infinite = vorrq_u64(visinfq_f64(x), visinfq_f64(y));
    let mut is_any_nan = vorrq_u64(visnanq_f64(x), visnanq_f64(y));
    let is_min_zero = vceqzq_f64(min);
    is_any_nan = vorrq_u64(visnanq_f64(ret), is_any_nan);
    ret = vbslq_f64(is_any_infinite, vdupq_n_f64(f64::INFINITY), ret);
    ret = vbslq_f64(is_any_nan, vdupq_n_f64(f64::NAN), ret);
    ret = vbslq_f64(is_min_zero, vdupq_n_f64(0f64), ret);
    ret
}
