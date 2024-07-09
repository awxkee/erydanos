/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{visinfq_f32, visnanq_f32, vmlafq_f32};
use std::arch::aarch64::*;

#[inline]
/// Method that computes 2D Euclidian distance *ULP 0.5*
pub unsafe fn vhypotq_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let max = vmaxq_f32(x, y);
    let min = vminq_f32(x, y);
    let r = vdivq_f32(min, max);
    let mut ret = vmulq_f32(vsqrtq_f32(vmlafq_f32(r, r, vdupq_n_f32(1f32))), max);
    let is_any_infinite = vandq_u32(visinfq_f32(x), visinfq_f32(y));
    let is_any_nan = vandq_u32(visnanq_f32(x), visnanq_f32(y));
    let is_min_zero = vceqzq_f32(min);
    let is_result_nan = visnanq_f32(ret);
    ret = vbslq_f32(is_any_infinite, vdupq_n_f32(f32::INFINITY), ret);
    ret = vbslq_f32(is_any_nan, vdupq_n_f32(f32::NAN), ret);
    ret = vbslq_f32(is_min_zero, vdupq_n_f32(0f32), ret);
    ret = vbslq_f32(is_result_nan, vdupq_n_f32(f32::INFINITY), ret);
    ret
}

/// Method that computes 2D Euclidian distance *ULP 0.5*, skipping Inf, Nan checks
#[inline]
pub unsafe fn vhypotq_fast_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    let x = vabsq_f32(x);
    let y = vabsq_f32(y);
    let max = vmaxq_f32(x, y);
    let min = vminq_f32(x, y);
    let r = vdivq_f32(min, max);
    let ret = vmulq_f32(vsqrtq_f32(vmlafq_f32(r, r, vdupq_n_f32(1f32))), max);
    ret
}
