/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::lnf::{LN_POLY_1_F, LN_POLY_2_F, LN_POLY_3_F, LN_POLY_4_F, LN_POLY_5_F};
use crate::neon::general::{vilogb2kq_f32, visinfq_f32, visnanq_f32, vldexp3kq_f32, vmlafq_f32};
use std::arch::aarch64::*;

/// Computes natural logarithm for an argument *ULP 1.5*
#[inline]
pub unsafe fn vlnq_f32(d: float32x4_t) -> float32x4_t {
    let mut res = vlnq_fast_f32(d);
    // d == 0 -> -Inf
    res = vbslq_f32(vceqzq_f32(d), vdupq_n_f32(f32::NEG_INFINITY), res);
    // d == Inf -> Inf
    res = vbslq_f32(visinfq_f32(d), vdupq_n_f32(f32::INFINITY), res);
    // d < 0 || d == Nan -> Nan
    res = vbslq_f32(
        vorrq_u32(vcltzq_f32(d), visnanq_f32(d)),
        vdupq_n_f32(f32::NAN),
        res,
    );
    res
}

/// Method that computes ln skipping Inf, Nan checks
#[inline]
pub unsafe fn vlnq_fast_f32(d: float32x4_t) -> float32x4_t {
    let n = vilogb2kq_f32(vmulq_n_f32(d, 1f32 / 0.75f32));
    let a = vldexp3kq_f32(d, vnegq_s32(n));
    let ones = vdupq_n_f32(1f32);
    let x = vdivq_f32(vsubq_f32(a, ones), vaddq_f32(a, ones));
    let x2 = vmulq_f32(x, x);
    let mut u = vdupq_n_f32(LN_POLY_5_F);
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_4_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_3_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_2_F));
    u = vmlafq_f32(u, x2, vdupq_n_f32(LN_POLY_1_F));
    let res = vmlafq_f32(
        vdupq_n_f32(std::f32::consts::LN_2),
        vcvtq_f32_s32(n),
        vmulq_f32(x, u),
    );
    res
}
