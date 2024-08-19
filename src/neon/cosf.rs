/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::cosf::{PI_A_F, PI_B_F, PI_C_F, PI_D_F};
use crate::neon::general::vmlafq_f32;
use crate::sinf::{SIN_POLY_1_S, SIN_POLY_2_S, SIN_POLY_3_S, SIN_POLY_4_S, SIN_POLY_5_S};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
/// Computes cosine function with error bound *ULP 1.5*
pub unsafe fn vcosq_f32(d: float32x4_t) -> float32x4_t {
    let half_1 = vdupq_n_f32(0.5f32);
    let q = vmlaq_s32(
        vdupq_n_s32(1),
        vcvtaq_s32_f32(vsubq_f32(
            vmulq_n_f32(d, std::f32::consts::FRAC_1_PI),
            half_1,
        )),
        vdupq_n_s32(2),
    );
    let qf = vcvtq_f32_s32(q);

    let mut r = vmlafq_f32(qf, vdupq_n_f32(-PI_A_F * 0.5), d);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_B_F * 0.5), r);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_C_F * 0.5), r);
    r = vmlafq_f32(qf, vdupq_n_f32(-PI_D_F * 0.5), r);

    let x2 = vmulq_f32(r, r);

    r = vbslq_f32(
        vceqq_u32(
            vandq_u32(vreinterpretq_u32_s32(q), vdupq_n_u32(2)),
            vdupq_n_u32(0),
        ),
        vnegq_f32(r),
        r,
    );
    let mut res = vdupq_n_f32(SIN_POLY_5_S);
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_4_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_3_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_2_S));
    res = vmlafq_f32(res, x2, vdupq_n_f32(SIN_POLY_1_S));
    res = vmlafq_f32(res, vmulq_f32(x2, r), r);
    res
}
