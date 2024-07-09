/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::expf::{
    EXP_POLY_1_S, EXP_POLY_2_S, EXP_POLY_3_S, EXP_POLY_4_S, EXP_POLY_5_S, L2L_F, L2U_F,
};
use crate::neon::general::{vmlafq_f32, vpow2ifq_s32};
use std::arch::aarch64::*;

/// Computes exp for an argument *ULP 1.0*
#[inline]
pub unsafe fn vexpq_f32(d: float32x4_t) -> float32x4_t {
    let mut r = vexpq_fast_f32(d);
    r = vbslq_f32(vcltq_f32(d, vdupq_n_f32(-87f32)), vdupq_n_f32(0f32), r);
    r = vbslq_f32(
        vcgtq_f32(d, vdupq_n_f32(88f32)),
        vdupq_n_f32(f32::INFINITY),
        r,
    );
    r
}

/// Method that computes exp skipping Inf, Nan checks error bound *ULP 1.0*
#[inline(always)]
pub unsafe fn vexpq_fast_f32(d: float32x4_t) -> float32x4_t {
    let q = vcvtaq_s32_f32(vmulq_n_f32(d, std::f32::consts::LOG2_E));
    let qf = vcvtq_f32_s32(q);
    /* exp(x) = 2^i * exp(f); */
    let mut r = vmlafq_f32(qf, vdupq_n_f32(-L2U_F), d);
    r = vmlafq_f32(qf, vdupq_n_f32(-L2L_F), r);
    let f = vmulq_f32(r, r);
    let mut u = vdupq_n_f32(EXP_POLY_5_S);
    u = vmlafq_f32(u, f, vdupq_n_f32(EXP_POLY_4_S));
    u = vmlafq_f32(u, f, vdupq_n_f32(EXP_POLY_3_S));
    u = vmlafq_f32(u, f, vdupq_n_f32(EXP_POLY_2_S));
    u = vmlafq_f32(u, f, vdupq_n_f32(EXP_POLY_1_S));
    let u = vaddq_f32(
        vdivq_f32(vmulq_n_f32(r, 2f32), vsubq_f32(u, r)),
        vdupq_n_f32(1f32),
    );
    let i2 = vreinterpretq_f32_s32(vpow2ifq_s32(q));
    let r = vmulq_f32(u, i2);
    r
}
