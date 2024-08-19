/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::neon::general::{vmlafq_f64, vmulq_s64};
use crate::sin::{
    PI_A2, PI_B2, SIN_POLY_10_D, SIN_POLY_2_D, SIN_POLY_3_D, SIN_POLY_4_D, SIN_POLY_5_D,
    SIN_POLY_6_D, SIN_POLY_7_D, SIN_POLY_8_D, SIN_POLY_9_D,
};

#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vcosq_f64(d: float64x2_t) -> float64x2_t {
    let half_1 = vdupq_n_f64(0.5f64);
    let pt = vcvtaq_s64_f64(vsubq_f64(
        vmulq_n_f64(d, std::f64::consts::FRAC_1_PI),
        half_1,
    ));
    let ml = vmulq_s64(pt, vdupq_n_s64(2));
    let q = vaddq_s64(vdupq_n_s64(1), ml);
    let qf = vcvtq_f64_s64(q);

    let mut r = vmlafq_f64(qf, vdupq_n_f64(-PI_A2 * 0.5), d);
    r = vmlafq_f64(qf, vdupq_n_f64(-PI_B2 * 0.5), r);

    let x2 = vmulq_f64(r, r);

    r = vbslq_f64(
        vceqq_u64(
            vandq_u64(vreinterpretq_u64_s64(q), vdupq_n_u64(2)),
            vdupq_n_u64(0),
        ),
        vnegq_f64(r),
        r,
    );
    let mut res = vdupq_n_f64(SIN_POLY_10_D);
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_9_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_8_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_7_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_6_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_5_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_4_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_3_D));
    res = vmlafq_f64(res, x2, vdupq_n_f64(SIN_POLY_2_D));
    res = vmlafq_f64(res, vmulq_f64(x2, r), r);
    res
}
