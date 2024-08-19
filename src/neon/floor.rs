/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{vcopysignq_f64, visinfq_f64};
use std::arch::aarch64::*;

/// Shorter and significantly faster reach skipping Inf checks
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vfloorq_f64(x: float64x2_t) -> float64x2_t {
    let ones = vdupq_n_f64(1f64);
    let z = vcvtq_s64_f64(x);
    let r = vcvtq_f64_s64(z);

    return vbslq_f64(vcgtq_f64(r, x), vsubq_f64(r, ones), r);
}

/// Floor method complaints with f64 specification with infinity checks
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vfloorq_ie_f64(x: float64x2_t) -> float64x2_t {
    let mut fr = vsubq_f64(x, vcvtq_f64_s64(vcvtq_s64_f64(x)));
    fr = vbslq_f64(vcltzq_f64(fr), vaddq_f64(fr, vdupq_n_f64(1f64)), fr);
    let c = vcopysignq_f64(vsubq_f64(x, fr), x);
    vbslq_f64(
        vorrq_u64(
            visinfq_f64(x),
            vcgeq_f64(vabsq_f64(x), vdupq_n_f64((1i64 << 52i64) as f64)),
        ),
        x,
        c,
    )
}
