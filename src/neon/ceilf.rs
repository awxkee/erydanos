/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::neon::general::{vcopysignq_f32, visinfq_f32};
use std::arch::aarch64::*;

/// Shorter and significantly faster reach skipping Inf checks
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vceilq_f32(x: float32x4_t) -> float32x4_t {
    let mut fr = vsubq_f32(x, vcvtq_f32_s32(vcvtq_s32_f32(x)));
    let ones = vdupq_n_f32(1f32);
    fr = vbslq_f32(vclezq_f32(fr), fr, vsubq_f32(fr, ones));
    vcopysignq_f32(vsubq_f32(x, fr), x)
}

/// Ceil's complaints with f32 specification with infinity checks
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn vceilq_ie_f32(x: float32x4_t) -> float32x4_t {
    let mut fr = vsubq_f32(x, vcvtq_f32_s32(vcvtq_s32_f32(x)));
    let ones = vdupq_n_f32(1f32);
    fr = vbslq_f32(vclezq_f32(fr), fr, vsubq_f32(fr, ones));
    let mask = vorrq_u32(visinfq_f32(x), vcgeq_f32(x, vdupq_n_f32((1 << 23) as f32)));
    let j2 = vcopysignq_f32(vsubq_f32(x, fr), x);
    vbslq_f32(mask, x, j2)
}
