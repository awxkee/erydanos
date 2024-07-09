/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::aarch64::*;

use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::neon::general::vmlafq_f32;
use crate::tanf::{
    TAN_POLY_1_S, TAN_POLY_2_S, TAN_POLY_3_S, TAN_POLY_4_S, TAN_POLY_5_S, TAN_POLY_6_S,
    TAN_POLY_7_S, TAN_POLY_8_S, TAN_POLY_9_S,
};

#[inline]
/// Computes tan with error bound *ULP 2.0*
pub unsafe fn vtanq_f32(d: float32x4_t) -> float32x4_t {
    let q = vcvtaq_s32_f32(vmulq_n_f32(d, std::f32::consts::FRAC_2_PI));
    let qf = vcvtq_f32_s32(q);

    let mut x = vmlafq_f32(qf, vdupq_n_f32(-PI_A_F * 0.5), d);
    x = vmlafq_f32(qf, vdupq_n_f32(-PI_B_F * 0.5), x);
    x = vmlafq_f32(qf, vdupq_n_f32(-PI_C_F * 0.5), x);
    x = vmlafq_f32(qf, vdupq_n_f32(-PI_D_F * 0.5), x);

    let even = vceqzq_s32(vandq_s32(q, vdupq_n_s32(1)));
    x = vbslq_f32(even, x, vnegq_f32(x));
    let x2 = vmulq_f32(x, x);
    let mut u = vdupq_n_f32(TAN_POLY_9_S);
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_8_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_7_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_6_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_5_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_4_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_3_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_2_S));
    u = vmlafq_f32(u, x2, vdupq_n_f32(TAN_POLY_1_S));
    u = vmlafq_f32(u, vmulq_f32(x2, x), x);
    u = vbslq_f32(even, u, vdivq_f32(vdupq_n_f32(1.), u));
    u
}
