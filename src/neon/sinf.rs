/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::neon::general::vmlafq_f32;
use crate::sinf::{SIN_POLY_1_S, SIN_POLY_2_S, SIN_POLY_3_S, SIN_POLY_4_S, SIN_POLY_5_S};

/// Computes sine function with error bound *ULP 1.5*
#[inline]
pub unsafe fn vsinq_f32(d: float32x4_t) -> float32x4_t {
    let q = vcvtaq_s32_f32(vmulq_n_f32(d, std::f32::consts::FRAC_1_PI));
    let qf = vcvtq_f32_s32(q);

    let mut r = vmlafq_f32(qf, vdupq_n_f32(-PI_A_F), d);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_B_F), r);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_C_F), r);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_D_F), r);

    let x2 = vmulq_f32(r, r);

    r = vbslq_f32(
        vceqq_u32(
            vandq_u32(vreinterpretq_u32_s32(q), vdupq_n_u32(1)),
            vdupq_n_u32(0),
        ),
        r,
        vnegq_f32(r),
    );
    let mut res = vdupq_n_f32(SIN_POLY_5_S);
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_4_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_3_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_2_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_1_S));
    res = vmlafq_f32(res, vmulq_f32(x2, r), r);
    res
}
