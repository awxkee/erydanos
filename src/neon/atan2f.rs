/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::atanf::vatanq_f32;
use std::arch::aarch64::*;

/// Computes atan for Y,X
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vatan2q_f32(y: float32x4_t, x: float32x4_t) -> float32x4_t {
    let zero_x_mask = vceqzq_f32(x);
    let yx = vatanq_f32(vdivq_f32(y, x));
    let mut rad = yx;
    rad = vbslq_f32(
        vandq_u32(zero_x_mask, vcgezq_f32(y)),
        vdupq_n_f32(std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y > 0.
    rad = vbslq_f32(
        vandq_u32(zero_x_mask, vclezq_f32(y)),
        vdupq_n_f32(-std::f32::consts::FRAC_PI_2),
        rad,
    ); // x == 0 && y < 0.
    rad = vbslq_f32(
        vandq_u32(zero_x_mask, vclezq_f32(y)),
        vdupq_n_f32(0f32),
        rad,
    ); // x == 0 && y == 0.
    let x_lower_than_0 = vcltzq_f32(x);
    rad = vbslq_f32(
        vandq_u32(x_lower_than_0, vcgezq_f32(y)),
        vaddq_f32(yx, vdupq_n_f32(std::f32::consts::PI)),
        rad,
    ); // x < 0 && y >= 0
    rad = vbslq_f32(
        vandq_u32(x_lower_than_0, vcltzq_f32(y)),
        vaddq_f32(yx, vdupq_n_f32(-std::f32::consts::PI)),
        rad,
    ); // x < 0 && y < 0
    rad
}
