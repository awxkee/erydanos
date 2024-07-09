/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{vcopysignq_f64, visinfq_f64};
use std::arch::aarch64::*;

/// Shorter and significantly faster reach skipping Inf checks
#[inline(always)]
pub unsafe fn vceilq_f64(x: float64x2_t) -> float64x2_t {
    let mut fr = vsubq_f64(x, vcvtq_f64_s64(vcvtq_s64_f64(x)));
    let ones = vdupq_n_f64(1f64);
    fr = vbslq_f64(vclezq_f64(fr), fr, vsubq_f64(fr, ones));
    vcopysignq_f64(vsubq_f64(x, fr), x)
}

/// Ceil's complaints with f64 specification with infinity checks
#[inline(always)]
pub unsafe fn vceilq_ie_f64(x: float64x2_t) -> float64x2_t {
    let mut fr = vsubq_f64(x, vcvtq_f64_s64(vcvtq_s64_f64(x)));
    let ones = vdupq_n_f64(1f64);
    fr = vbslq_f64(vclezq_f64(fr), fr, vsubq_f64(fr, ones));
    let mask = vorrq_u64(
        visinfq_f64(x),
        vcgeq_f64(x, vdupq_n_f64((1i64 << 52i64) as f64)),
    );
    let j2 = vcopysignq_f64(vsubq_f64(x, fr), x);
    vbslq_f64(mask, x, j2)
}
