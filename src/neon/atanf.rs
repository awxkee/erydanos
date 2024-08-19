/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::atanf::{
    ATAN_POLY_1_F, ATAN_POLY_2_F, ATAN_POLY_3_F, ATAN_POLY_4_F, ATAN_POLY_5_F, ATAN_POLY_6_F,
    ATAN_POLY_7_F, ATAN_POLY_8_F, ATAN_POLY_9_F,
};
use crate::neon::general::vmlafq_f32;
use std::arch::aarch64::*;

/// Computes Atan function with *ULP 1.0* error
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vatanq_f32(x: float32x4_t) -> float32x4_t {
    let negative_mask = vcltzq_f32(x);
    let d = vabsq_f32(x);
    let more_than_one_mask = vcgeq_f32(d, vdupq_n_f32(1f32));
    let x = vbslq_f32(more_than_one_mask, vdivq_f32(vdupq_n_f32(1f32), d), d);
    let x2 = vmulq_f32(x, x);
    let mut u = vdupq_n_f32(ATAN_POLY_9_F);
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_8_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_7_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_6_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_5_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_4_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_3_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_2_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(ATAN_POLY_1_F));
    u = vmulq_f32(u, x);
    u = vbslq_f32(
        more_than_one_mask,
        vsubq_f32(vdupq_n_f32(std::f32::consts::FRAC_PI_2), u),
        u,
    );
    u = vbslq_f32(negative_mask, vnegq_f32(u), u);
    u
}
